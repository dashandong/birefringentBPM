%% Simulated full Mueller matrix measurement based on Multislice 3D Beam-Propagation
% Copyright (C) 2024 by Dr. Dashan Dong (dongdashan@icloud.com)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% Last Modified: 2024/10/31

% Please refer to my paper for more details:
% Shuqi Mu, Yingtong Shi, Yintong Song, Wei Liu, Wanxue Wei, Qihuang
% Gong, Dashan Dong*, and Kebin Shi, "Multi-slice computational
% model for birefringent scattering", Optica 10.1, 2023.

function fullMueller
    %% Physical Parameters
    n_medium = 1.44; % Refractive Index of medium
    lambda = 405.0e-9; % Wavelength
    k0 = 2 * pi / lambda; % Wavenumber
    k_m = n_medium * k0; % Maximum wavenumber
    NA = 0.8; % Numerical Aperture

    %% Demensions of the simulation
    N_x = 180;
    N_y = 180;
    N_pad = 38; % Padding size around the field

    d_x = 65e-9;
    d_y = 65e-9;

    d_z = 65e-9; % Pixel Size in Z direction
    z_o = -2.275e-6; % Distance between Field_origin_z & Field_input_z
    z_e = 2.275e-6 + d_z; % Distance between Field_input_z & Field_end_z

    % Generate the grids
    [N_z, ax_x, ax_y, ax_z, grid2_X, grid2_Y, grid3_X, grid3_Y, grid3_Z, padMask] = makeGrids(N_x, N_y, N_pad, d_x, d_y, z_o, z_e, d_z);

    % Generate the k-space grids
    [ax_Kx, ax_Ky, grid2_Kx, grid2_Ky, grid2_Kr] = makeKGrids(N_x, N_y, N_pad, d_x, d_y);

    % Scattering Potential of the 4 beads in Figure 2
    [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz] = scatteringPotential_beads(grid3_X, grid3_Y, grid3_Z);

    % Generate PTFT tensor
    [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz] = makePTFT();

    %% Some pre-calculations for faster computation
    % Pre ifftshift the PTFT tensor for faster computation
    [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz] = deal(ifftshift(Qxx), ifftshift(Qyy), ifftshift(Qzz), ifftshift(Qxy), ifftshift(Qxz), ifftshift(Qyz));
    % Pre ifftshift the scattering potential tensor for faster computation
    % do this in both dimension 1 & 2
    [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz] = deal(ifftshift(Vxx, 1), ifftshift(Vyy, 1), ifftshift(Vzz, 1), ifftshift(Vxy, 1), ifftshift(Vxz, 1), ifftshift(Vyz, 1));
    [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz] = deal(ifftshift(Vxx, 2), ifftshift(Vyy, 2), ifftshift(Vzz, 2), ifftshift(Vxy, 2), ifftshift(Vxz, 2), ifftshift(Vyz, 2));
    % Pre ifftshift the padding mask for faster computation
    padMask = ifftshift(padMask);
    % Free-space propagation kernel detialed in Eq.(S25)
    [PKxx, PKyy, PKzz, PKxy, PKxz, PKyz] = freeSpaceKernal(Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, d_z);
    % Vectorial scattering kernel detialed in Eq.(S35)
    [HKxx, HKyy, HKzz, HKxy, HKxz, HKyz] = birefringentScatteringKernel(Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, d_z);
    % Lowpass filter with NA limite
    naLPFilter = makeNAFilter(NA);

    %% We measure the full Mueller matrix by 6 input polarization states
    % Jones vector of the input field
    inputJones = [1, 0; ... $ Linear Horizontal
                      0, 1; ... $ Linear Vertical
                      1, 1; ... $ Linear 45 degree
                      1, -1; ... $ Linear -45 degree
                      1, 1i; ... $ Right Circular
                      1, -1i]; % Left Circular

    % Results save in the following stokes vectors
    S0 = zeros(N_x + 2 * N_pad, N_y + 2 * N_pad, 6);
    S1 = zeros(N_x + 2 * N_pad, N_y + 2 * N_pad, 6);
    S2 = zeros(N_x + 2 * N_pad, N_y + 2 * N_pad, 6);
    S3 = zeros(N_x + 2 * N_pad, N_y + 2 * N_pad, 6);

    for i = 1:6
        % Generate the input field
        [Ux, Uy, Uz] = makeInputField(inputJones(i, :), 0, 0);
        [hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase] = plotField(Ux, Uy, Uz, ax_x, ax_y);
        % Pre ifftshift the field for faster computation
        [Ux, Uy, Uz] = deal(ifftshift(Ux), ifftshift(Uy), ifftshift(Uz));

        % Multislice Propagation
        for j = 1:(N_z - 1)
            %% The field at n+1 slice is the sum of two parts: (using Eq.(8) in paper)
            % 1. The field at n slice after propagation (calculate using Eq.(12) in paper)
            % 2. The field at n slice after scattering (calculate using Eq.(14) in paper)

            %% Part 1: Vectorial Free-space Propagation
            % Detialed in Supplementary Note 2
            % Using Eqaution (S24 & S25) in paper
            [Ux_free, Uy_free, Uz_free] = freeSpaceVPropagation(Ux, Uy, Uz);

            %% Part 2: Birefringent Scattering
            % Detialed in Supplementary Note 3
            % Using Eqaution (S33 & S34 & S35) in paper
            [Ux_scatter, Uy_scatter, Uz_scatter] = birefringentScattering(Ux, Uy, Uz, j);

            % Update the field at n+1 slice
            Ux = Ux_free + Ux_scatter;
            Uy = Uy_free + Uy_scatter;
            Uz = Uz_free + Uz_scatter;

            % Cancel the propagation phase
            Ux = Ux .* exp(-1i * k0 * n_medium * d_z);
            Uy = Uy .* exp(-1i * k0 * n_medium * d_z);
            Uz = Uz .* exp(-1i * k0 * n_medium * d_z);
            updateFieldPlot(j + 1, hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase, Ux .* padMask, Uy .* padMask, Uz .* padMask);
        end

        %% Free-space Backward Propagation to Focal Plane
        for j = (N_z - 1):-1:(floor(N_z / 2) + 1)
            [Ux, Uy, Uz] = freeSpaceVBackPropagation(Ux, Uy, Uz);
            Ux = Ux .* exp(+1i * k0 * n_medium * d_z);
            Uy = Uy .* exp(+1i * k0 * n_medium * d_z);
            Uz = Uz .* exp(+1i * k0 * n_medium * d_z);
            updateFieldPlot(j, hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase, Ux .* padMask, Uy .* padMask, Uz .* padMask);
        end

        % Measure the Stokes parameters
        S0(:, :, i) = Ux .* conj(Ux) + Uy .* conj(Uy) + Uz .* conj(Uz);
        S1(:, :, i) = Ux .* conj(Ux) - Uy .* conj(Uy);
        S2(:, :, i) = Ux .* conj(Uy) + Uy .* conj(Ux);
        S3(:, :, i) = 1i * (Ux .* conj(Uy) - Uy .* conj(Ux));
    end

    M = zeros(N_x + 2 * N_pad, N_y + 2 * N_pad, 4, 4);
    M(:, :, 1, 1) = (S0(:, :, 1) + S0(:, :, 2) + S0(:, :, 3) + S0(:, :, 4) + S0(:, :, 5) + S0(:, :, 6)) / 6;
    M(:, :, 1, 2) = (S0(:, :, 1) - S0(:, :, 2)) / 2;
    M(:, :, 1, 3) = (S0(:, :, 3) - S0(:, :, 4)) / 2;
    M(:, :, 1, 4) = (S0(:, :, 5) - S0(:, :, 6)) / 2;

    M(:, :, 2, 1) = (S1(:, :, 1) + S1(:, :, 2) + S1(:, :, 3) + S1(:, :, 4) + S1(:, :, 5) + S1(:, :, 6)) / 6;
    M(:, :, 2, 2) = (S1(:, :, 1) - S1(:, :, 2)) / 2;
    M(:, :, 2, 3) = (S1(:, :, 3) - S1(:, :, 4)) / 2;
    M(:, :, 2, 4) = (S1(:, :, 5) - S1(:, :, 6)) / 2;

    M(:, :, 3, 1) = (S2(:, :, 1) + S2(:, :, 2) + S2(:, :, 3) + S2(:, :, 4) + S2(:, :, 5) + S2(:, :, 6)) / 6;
    M(:, :, 3, 2) = (S2(:, :, 1) - S2(:, :, 2)) / 2;
    M(:, :, 3, 3) = (S2(:, :, 3) - S2(:, :, 4)) / 2;
    M(:, :, 3, 4) = (S2(:, :, 5) - S2(:, :, 6)) / 2;

    M(:, :, 4, 1) = (S3(:, :, 1) + S3(:, :, 2) + S3(:, :, 3) + S3(:, :, 4) + S3(:, :, 5) + S3(:, :, 6)) / 6;
    M(:, :, 4, 2) = (S3(:, :, 1) - S3(:, :, 2)) / 2;
    M(:, :, 4, 3) = (S3(:, :, 3) - S3(:, :, 4)) / 2;
    M(:, :, 4, 4) = (S3(:, :, 5) - S3(:, :, 6)) / 2;

    % fftshift the Mueller matrix for better visualization
    M = fftshift(M, 1);
    M = fftshift(M, 2);

    % Plot the Mueller matrix
    figure;
    tiledlayout(4, 4);

    for i = 1:4

        for j = 1:4
            nexttile;
            imagesc(ax_x * 1e6, ax_y * 1e6, squeeze(M(:, :, i, j))');
            axis xy image;
            xlim([ax_x(N_pad + 1), ax_x(N_pad + N_x)] * 1e6);
            ylim([ax_y(N_pad + 1), ax_x(N_pad + N_y)] * 1e6);
            title(['M_{', num2str(i), num2str(j), '}']);
            colormap(gca, 'jet');
            clim([-2, 2]);
            xlabel('x (\mum)');
            ylabel('y (\mum)');
            cb = colorbar;
            cb.Location = 'eastoutside';
        end

    end

    %% Helper Functions

    function [Ux_free, Uy_free, Uz_free] = freeSpaceVPropagation(U_x, U_y, U_z)
        % Detialed in Supplementary Note 2
        % Using Eqaution (S24) in paper
        % ifftshift first so the phase reference is at the center of the field
        Ux_FT = fft2(U_x);
        Uy_FT = fft2(U_y);
        Uz_FT = fft2(U_z);
        Ux_free = naLPFilter .* (Ux_FT .* PKxx + Uy_FT .* PKxy + Uz_FT .* PKxz);
        Uy_free = naLPFilter .* (Ux_FT .* PKxy + Uy_FT .* PKyy + Uz_FT .* PKyz);
        Uz_free = naLPFilter .* (Ux_FT .* PKxz + Uy_FT .* PKyz + Uz_FT .* PKzz);
        Ux_free = ifft2(Ux_free);
        Uy_free = ifft2(Uy_free);
        Uz_free = ifft2(Uz_free);
    end

    function [Ux_free, Uy_free, Uz_free] = freeSpaceVBackPropagation(U_x, U_y, U_z)
        % Detialed in Supplementary Note 2
        % Using Eqaution (S24) in paper
        % ifftshift first so the phase reference is at the center of the field
        Ux_FT = fft2(U_x);
        Uy_FT = fft2(U_y);
        Uz_FT = fft2(U_z);
        Ux_free = naLPFilter .* (Ux_FT .* conj(PKxx) + Uy_FT .* conj(PKxy) + Uz_FT .* conj(PKxz));
        Uy_free = naLPFilter .* (Ux_FT .* conj(PKxy) + Uy_FT .* conj(PKyy) + Uz_FT .* conj(PKyz));
        Uz_free = naLPFilter .* (Ux_FT .* conj(PKxz) + Uy_FT .* conj(PKyz) + Uz_FT .* conj(PKzz));
        Ux_free = ifft2(Ux_free);
        Uy_free = ifft2(Uy_free);
        Uz_free = ifft2(Uz_free);
    end

    function [Ux_scatter, Uy_scatter, Uz_scatter] = birefringentScattering(U_x, U_y, U_z, i)
        % Detialed in Supplementary Note 3
        % Using Eqaution (S33 & S34 & S35) in paper

        % First, go to k-space and decompose to dipole components in current slice
        Ux_FT = fft2(U_x);
        Uy_FT = fft2(U_y);
        Uz_FT = fft2(U_z);
        Ux_Dc = naLPFilter .* (Qxx .* Ux_FT + Qxy .* Uy_FT + Qxz .* Uz_FT);
        Uy_Dc = naLPFilter .* (Qxy .* Ux_FT + Qyy .* Uy_FT + Qyz .* Uz_FT);
        Uz_Dc = naLPFilter .* (Qxz .* Ux_FT + Qyz .* Uy_FT + Qzz .* Uz_FT);
        Ux_Dc = ifft2(Ux_Dc);
        Uy_Dc = ifft2(Uy_Dc);
        Uz_Dc = ifft2(Uz_Dc);

        % Second, back to spatial domain and apply the scattering potential
        Ux_Sc = Vxx(:, :, i) .* Ux_Dc + Vxy(:, :, i) .* Uy_Dc + Vxz(:, :, i) .* Uz_Dc;
        Uy_Sc = Vxy(:, :, i) .* Ux_Dc + Vyy(:, :, i) .* Uy_Dc + Vyz(:, :, i) .* Uz_Dc;
        Uz_Sc = Vxz(:, :, i) .* Ux_Dc + Vyz(:, :, i) .* Uy_Dc + Vzz(:, :, i) .* Uz_Dc;

        % Third, go to k-space again and z integrate the scattering components to the second slice
        Ux_Sc = fft2(Ux_Sc);
        Uy_Sc = fft2(Uy_Sc);
        Uz_Sc = fft2(Uz_Sc);
        Ux_scatter = naLPFilter .* (HKxx .* Ux_Sc + HKxy .* Uy_Sc + HKxz .* Uz_Sc);
        Uy_scatter = naLPFilter .* (HKxy .* Ux_Sc + HKyy .* Uy_Sc + HKyz .* Uz_Sc);
        Uz_scatter = naLPFilter .* (HKxz .* Ux_Sc + HKyz .* Uy_Sc + HKzz .* Uz_Sc);
        Ux_scatter = ifft2(Ux_scatter);
        Uy_scatter = ifft2(Uy_scatter);
        Uz_scatter = ifft2(Uz_scatter);

    end

    function [PK_xx, PK_yy, PK_zz, PK_xy, PK_xz, PK_yz] = freeSpaceKernal(Q_xx, Q_yy, Q_zz, Q_xy, Q_xz, Q_yz, d_z)
        % Detialed in Supplementary Note 2
        % Using Eqaution (S25) in paper
        % The kernel is pre-fftshifted for faster computation
        KrShift = ifftshift(grid2_Kr);
        kzSquare = k_m ^ 2 - KrShift .^ 2;
        kzSquare(kzSquare < 0) = 0;
        Kz = sqrt(kzSquare);
        LPFilter = KrShift < (k_m);

        PK_xx = LPFilter .* complex(Q_xx) .* exp(1i * Kz * d_z);
        PK_yy = LPFilter .* complex(Q_yy) .* exp(1i * Kz * d_z);
        PK_zz = LPFilter .* complex(Q_zz) .* exp(1i * Kz * d_z);
        PK_xy = LPFilter .* complex(Q_xy) .* exp(1i * Kz * d_z);
        PK_xz = LPFilter .* complex(Q_xz) .* exp(1i * Kz * d_z);
        PK_yz = LPFilter .* complex(Q_yz) .* exp(1i * Kz * d_z);
    end

    function [HK_xx, HK_yy, HK_zz, HK_xy, HK_xz, HK_yz] = birefringentScatteringKernel(Q_xx, Q_yy, Q_zz, Q_xy, Q_xz, Q_yz, d_z)
        % Detialed in Supplementary Note 3
        % Using Eqaution (S35) in paper
        % The kernel is pre-fftshifted for faster computation
        KrShift = ifftshift(grid2_Kr);
        kzSquare = k_m ^ 2 - KrShift .^ 2;
        kzSquare(kzSquare < 0) = 0;
        Kz = sqrt(kzSquare);
        Kz(Kz < eps) = eps;
        LPFilter = KrShift < (k_m);

        HK_xx = LPFilter .* (-1i ./ 2 .* complex(Q_xx) .* exp(1i * Kz * d_z) ./ Kz .* d_z);
        HK_yy = LPFilter .* (-1i ./ 2 .* complex(Q_yy) .* exp(1i * Kz * d_z) ./ Kz .* d_z);
        HK_zz = LPFilter .* (-1i ./ 2 .* complex(Q_zz) .* exp(1i * Kz * d_z) ./ Kz .* d_z);
        HK_xy = LPFilter .* (-1i ./ 2 .* complex(Q_xy) .* exp(1i * Kz * d_z) ./ Kz .* d_z);
        HK_xz = LPFilter .* (-1i ./ 2 .* complex(Q_xz) .* exp(1i * Kz * d_z) ./ Kz .* d_z);
        HK_yz = LPFilter .* (-1i ./ 2 .* complex(Q_yz) .* exp(1i * Kz * d_z) ./ Kz .* d_z);
    end

    function lpFilter = makeNAFilter(NA)
        % NA filter
        % The filter is pre-fftshifted for faster computation
        KrShift = ifftshift(grid2_Kr);
        lpFilter = KrShift < (k0 * NA);
    end

    function updateFieldPlot(i, hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase, Ux, Uy, Uz)
        hUx_abs.CData = abs(fftshift(Ux))';
        hUy_abs.CData = abs(fftshift(Uy))';
        hUz_abs.CData = abs(fftshift(Uz))';
        hUx_phase.CData = angle(fftshift(Ux))';
        hUy_phase.CData = angle(fftshift(Uy))';
        hUz_phase.CData = angle(fftshift(Uz))';
        hFig.Name = ['Field at z = ', num2str(ax_z(i) * 1e6), ' um'];
        drawnow;
    end

    function [N_z, aX, aY, aZ, grid2X, grid2Y, gridX, gridY, gridZ, maskPad] = makeGrids(N_x, N_y, N_pad, d_x, d_y, z_o, z_e, d_z)
        L_x = d_x * (N_x + 2 * N_pad); % X_Length of the field
        aX = linspace(-L_x / 2, L_x / 2, (N_x + 2 * N_pad));

        L_y = d_y * (N_y + 2 * N_pad); % Y_Length of the field
        aY = linspace(-L_y / 2, L_y / 2, (N_y + 2 * N_pad));

        [grid2Y, grid2X] = meshgrid(aY, aX);

        maskPad = zeros(N_x + 2 * N_pad, N_y + 2 * N_pad);
        maskPad(N_pad + 1:end - N_pad, N_pad + 1:end - N_pad) = 1;

        L_z = z_e - z_o; % Z_Length of the field
        N_z = floor(L_z / d_z) + 1;
        aZ = linspace(z_o, z_o + d_z * (N_z - 1), N_z);

        [gridY, gridX, gridZ] = meshgrid(aY, aX, aZ);
    end

    function [k_x, k_y, K_x, K_y, K_r] = makeKGrids(N_x, N_y, N_pad, d_x, d_y)
        dk_x = 2 * pi / (d_x * (N_x + 2 * N_pad));
        dk_y = 2 * pi / (d_y * (N_y + 2 * N_pad));

        k_x = dk_x .* ((1:1:(N_x + 2 * N_pad)) - ceil((N_x + 2 * N_pad + 1) / 2));
        k_y = dk_y .* ((1:1:(N_y + 2 * N_pad)) - ceil((N_y + 2 * N_pad + 1) / 2));

        [K_y, K_x] = meshgrid(k_y, k_x);
        [~, K_r] = cart2pol(K_x, K_y);
    end

    function [U_x, U_y, U_z] = makeInputField(vJones, azDeg, elDeg)
        vJones = vJones ./ norm(vJones);

        % Horizontal k vector
        k_x = k_m * sind(elDeg) * cosd(azDeg);
        k_y = k_m * sind(elDeg) * sind(azDeg);

        % Fix the diffraction from boundary by rounding the k vector
        dk_x = 2 * pi / (d_x * (N_x + 2 * N_pad));
        dk_y = 2 * pi / (d_y * (N_y + 2 * N_pad));
        k_x = round(k_x / dk_x) * dk_x;
        k_y = round(k_y / dk_y) * dk_y;

        % Complex Amplitude of the input field
        A = [vJones(1); vJones(2); 0];

        % Rotate the input field
        M = [1 + (cosd(elDeg) - 1) * cosd(azDeg) ^ 2, (cosd(elDeg) - 1) * cosd(azDeg) * sind(azDeg), -sind(elDeg) * cosd(azDeg); ...
                 (cosd(elDeg) - 1) * cosd(azDeg) * sind(azDeg), 1 + (cosd(elDeg) - 1) * sind(azDeg) ^ 2, -sind(elDeg) * sind(azDeg); ...
                 sind(elDeg) * cosd(azDeg), sind(elDeg) * sind(azDeg), cosd(elDeg)];
        A = M * A;

        % Generate the input field
        U_x = A(1) * exp(1i * (k_x * grid2_X + k_y * grid2_Y));
        U_y = A(2) * exp(1i * (k_x * grid2_X + k_y * grid2_Y));
        U_z = A(3) * exp(1i * (k_x * grid2_X + k_y * grid2_Y));

        % For Gaussian Beam
        % w0 = 10e-6;
        % U_x = U_x .* exp(-grid2_X .^ 2 / w0 ^ 2);
        % U_y = U_y .* exp(-grid2_Y .^ 2 / w0 ^ 2);
        % U_z = U_z .* exp(-grid2_X .^ 2 / w0 ^ 2);

    end

    function [Q_xx, Q_yy, Q_zz, Q_xy, Q_xz, Q_yz] = makePTFT()
        kzSquare = k_m ^ 2 - grid2_Kr .^ 2;
        clipping = kzSquare <= 0; % Clipping the evanescent waves
        kzSquare(clipping) = 0;
        LPFilter = ~clipping;

        % PTFT tensor, Eq.(7) in paper
        Q_xx = LPFilter .* (1 - grid2_Kx .^ 2 ./ k_m .^ 2);
        Q_yy = LPFilter .* (1 - grid2_Ky .^ 2 ./ k_m .^ 2);
        Q_zz = LPFilter .* (1 - kzSquare ./ k_m .^ 2);
        Q_xy = LPFilter .* (-grid2_Kx .* grid2_Ky ./ k_m .^ 2);
        Q_xz = LPFilter .* (-grid2_Kx .* sqrt(kzSquare) ./ k_m .^ 2);
        Q_yz = LPFilter .* (-grid2_Ky .* sqrt(kzSquare) ./ k_m .^ 2);
    end

    function [hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase] = plotField(Ux, Uy, Uz, ax_x, ax_y)
        hFig = figure('Name', ['Field at z = ', num2str(z_o * 1e6), ' um']);
        tiledlayout(hFig, 2, 3);
        nexttile;
        hUx_abs = imagesc(ax_x * 1e6, ax_y * 1e6, abs(Ux'));
        axis xy image;
        xlim([ax_x(N_pad + 1), ax_x(N_pad + N_x)] * 1e6);
        ylim([ax_y(N_pad + 1), ax_x(N_pad + N_y)] * 1e6);
        title('U_x Amplitude');
        % clim([0, 2.5]);
        colormap(gca, 'hot');
        ylabel('y (\mum)');
        cb_Abs = colorbar;
        cb_Abs.Location = 'eastoutside';
        nexttile;
        hUy_abs = imagesc(ax_x * 1e6, ax_y * 1e6, abs(Uy'));
        axis xy image;
        xlim([ax_x(N_pad + 1), ax_x(N_pad + N_x)] * 1e6);
        ylim([ax_y(N_pad + 1), ax_x(N_pad + N_y)] * 1e6);
        title('U_y Amplitude');
        % clim([0, 2.5]);
        colormap(gca, 'hot');
        xlabel('x (\mum)');
        cb_Abs = colorbar;
        cb_Abs.Location = 'eastoutside';
        nexttile;
        hUz_abs = imagesc(ax_x * 1e6, ax_y * 1e6, abs(Uz'));
        axis xy image;
        xlim([ax_x(N_pad + 1), ax_x(N_pad + N_x)] * 1e6);
        ylim([ax_y(N_pad + 1), ax_x(N_pad + N_y)] * 1e6);
        title('U_z Amplitude');
        % clim([0, 1.0]);
        colormap(gca, 'hot');
        cb_Abs = colorbar;
        cb_Abs.Location = 'eastoutside';

        nexttile;
        hUx_phase = imagesc(ax_x * 1e6, ax_y * 1e6, angle(Ux'));
        axis xy image;
        xlim([ax_x(N_pad + 1), ax_x(N_pad + N_x)] * 1e6);
        ylim([ax_y(N_pad + 1), ax_x(N_pad + N_y)] * 1e6);
        title('U_x Phase');
        clim([-pi, pi]);
        colormap(gca, 'parula');
        ylabel('y (\mum)');
        nexttile;
        hUy_phase = imagesc(ax_x * 1e6, ax_y * 1e6, angle(Uy'));
        axis xy image;
        xlim([ax_x(N_pad + 1), ax_x(N_pad + N_x)] * 1e6);
        ylim([ax_y(N_pad + 1), ax_x(N_pad + N_y)] * 1e6);
        title('U_y Phase');
        clim([-pi, pi]);
        colormap(gca, 'parula');
        xlabel('x (\mum)');
        nexttile;
        hUz_phase = imagesc(ax_x * 1e6, ax_y * 1e6, angle(Uz'));
        axis xy image;
        xlim([ax_x(N_pad + 1), ax_x(N_pad + N_x)] * 1e6);
        ylim([ax_y(N_pad + 1), ax_x(N_pad + N_y)] * 1e6);
        title('U_z Phase');
        clim([-pi, pi]);
        colormap(gca, 'parula');
        cb_Phase = colorbar;
        cb_Phase.Location = 'eastoutside';

        drawnow;
    end

    function [V_xx, V_yy, V_zz, V_xy, V_xz, V_yz] = scatteringPotential_beads(grid3_X, grid3_Y, grid3_Z)
        V_xx = zeros(size(grid3_X));
        V_yy = zeros(size(grid3_X));
        V_zz = zeros(size(grid3_X));
        V_xy = zeros(size(grid3_X));
        V_xz = zeros(size(grid3_X));
        V_yz = zeros(size(grid3_X));

        % Fill the particle region with rotated scattering potential
        % Dn for vaterite is ~0.06 accroding to Ref[42]
        n_beads_xx = 1.48; % Refractive Index of sphere particle
        n_beads_yy = 1.54;
        n_beads_zz = 1.54;
        particle_radius_x = 2e-6; % Radius of sphere particle
        particle_radius_y = 3e-6;
        particle_radius_z = 2e-6;

        epsilon = [n_beads_xx ^ 2, 0, 0; 0, n_beads_yy ^ 2, 0; 0, 0, n_beads_zz ^ 2];
        delta_epsilon = [n_medium ^ 2, 0, 0; 0, n_medium ^ 2, 0; 0, 0, n_medium ^ 2] - epsilon;
        particle_Coord_R = (grid3_X ./ particle_radius_x) .^ 2 + (grid3_Y ./ particle_radius_y) .^ 2 + (grid3_Z ./ particle_radius_z) .^ 2 < 1;
        V_xx(particle_Coord_R) = k0 .^ 2 * delta_epsilon(1, 1);
        V_yy(particle_Coord_R) = k0 .^ 2 * delta_epsilon(2, 2);
        V_zz(particle_Coord_R) = k0 .^ 2 * delta_epsilon(3, 3);
        V_xy(particle_Coord_R) = k0 .^ 2 * delta_epsilon(1, 2);
        V_xz(particle_Coord_R) = k0 .^ 2 * delta_epsilon(1, 3);
        V_yz(particle_Coord_R) = k0 .^ 2 * delta_epsilon(2, 3);

    end

end
