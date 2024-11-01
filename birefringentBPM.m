% Multislice 3D Beam-Propagation Simulation for Birefringent Scattering
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
% Last Modified: 2024/10/28

% This script is used to simulate the vectorial beam propagation
% with birefringent scattering in a 3D volume. The simulation is
% based on the multislice method and the fast Fourier transform
% beam propagation method.
%
% Please refer to my paper for more details:
% Shuqi Mu, Yingtong Shi, Yintong Song, Wei Liu, Wanxue Wei, Qihuang
% Gong, Dashan Dong*, and Kebin Shi, "Multi-slice computational
% model for birefringent scattering", Optica 10.1, 2023.

function birefringentBPM()

    %% Physical Parameters
    n_medium = 1.33; % Refractive Index of medium
    lambda = 405.0e-9; % Wavelength
    k0 = 2 * pi / lambda; % Wavenumber
    k_m = n_medium * k0; % Maximum wavenumber
    NA = 1.32; % Numerical Aperture

    %% Demensions of the simulation
    N_x = 180;
    N_y = 180;
    N_pad = 38; % Padding size around the field

    d_x = 65e-9;
    d_y = 65e-9;

    d_z = 65e-9; % Pixel Size in Z direction
    z_o = -2.275e-6; % Distance between Field_origin_z & Field_input_z
    z_e = 2.275e-6; % Distance between Field_input_z & Field_end_z

    % Generate the grids
    [N_z, ax_x, ax_y, ax_z, grid2_X, grid2_Y, grid3_X, grid3_Y, grid3_Z, padMask] = makeGrids(N_x, N_y, N_pad, d_x, d_y, z_o, z_e, d_z);

    % Generate the k-space grids
    [ax_Kx, ax_Ky, grid2_Kx, grid2_Ky, grid2_Kr] = makeKGrids(N_x, N_y, N_pad, d_x, d_y);

    % Scattering Potential of the 4 beads in Figure 2
    [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz] = scatteringPotential_beads(grid3_X, grid3_Y, grid3_Z);

    % Generate the input field
    [Ux, Uy, Uz] = makeInputField([1, 0], 0, 0);
    % [Ux, Uy, Uz] = makeInputField([1, 0], 225, 36.87);

    % Generate PTFT tensor
    [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz] = makePTFT();

    % Plot the field at input
    [hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase] = plotField(Ux, Uy, Uz, ax_x, ax_y);

    %% Some pre-calculations for faster computation
    % Pre ifftshift the PTFT tensor for faster computation
    [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz] = deal(ifftshift(Qxx), ifftshift(Qyy), ifftshift(Qzz), ifftshift(Qxy), ifftshift(Qxz), ifftshift(Qyz));
    % Pre ifftshift the scattering potential tensor for faster computation
    % do this in both dimension 1 & 2
    [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz] = deal(ifftshift(Vxx, 1), ifftshift(Vyy, 1), ifftshift(Vzz, 1), ifftshift(Vxy, 1), ifftshift(Vxz, 1), ifftshift(Vyz, 1));
    [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz] = deal(ifftshift(Vxx, 2), ifftshift(Vyy, 2), ifftshift(Vzz, 2), ifftshift(Vxy, 2), ifftshift(Vxz, 2), ifftshift(Vyz, 2));
    % Pre ifftshift the field for faster computation
    [Ux, Uy, Uz] = deal(ifftshift(Ux), ifftshift(Uy), ifftshift(Uz));
    % Pre ifftshift the padding mask for faster computation
    padMask = ifftshift(padMask);
    % Free-space propagation kernel detialed in Eq.(S25)
    [PKxx, PKyy, PKzz, PKxy, PKxz, PKyz] = freeSpaceKernal(Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, d_z);
    % Vectorial scattering kernel detialed in Eq.(S35)
    [HKxx, HKyy, HKzz, HKxy, HKxz, HKyz] = birefringentScatteringKernel(Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, d_z);
    % Lowpass filter with NA limite
    naLPFilter = makeNAFilter(NA);

    % Multislice Propagation
    for i = 1:(N_z - 1)
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
        [Ux_scatter, Uy_scatter, Uz_scatter] = birefringentScattering(Ux, Uy, Uz, i);

        % Update the field at n+1 slice
        Ux = Ux_free + Ux_scatter;
        Uy = Uy_free + Uy_scatter;
        Uz = Uz_free + Uz_scatter;

        % Cancel the propagation phase
        Ux = Ux .* exp(-1i * k0 * n_medium * d_z);
        Uy = Uy .* exp(-1i * k0 * n_medium * d_z);
        Uz = Uz .* exp(-1i * k0 * n_medium * d_z);

        % Plot the field at n+1 slice
        updateFieldPlot(i + 1, hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase, Ux .* padMask, Uy .* padMask, Uz .* padMask);
    end

    %% Vectorial Beam Propagation Method for comparison
    % Generate the input field
    [Ux, Uy, Uz] = makeInputField([1, 0], 0, 0);
    % [Ux, Uy, Uz] = makeInputField([1, 0], 225, 36.87);

    % Pre ifftshift the field for faster computation
    [Ux, Uy, Uz] = deal(ifftshift(Ux), ifftshift(Uy), ifftshift(Uz));
    % Plot the field at input
    [hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase] = plotField(Ux, Uy, Uz, ax_x, ax_y);
    % Pre-calculate the BPM kernel
    BK = vBPMKernal(d_z);
    % Convert the scattering potential to modulation matrix
    [Phixx, Phiyy, Phizz, Phixy, Phixz, Phiyz] = phaseModulation_beads(grid3_X, grid3_Y, grid3_Z);
    % Pre ifftshift the modulation matrix for faster computation
    % do this in both dimension 1 & 2
    [Phixx, Phiyy, Phizz, Phixy, Phixz, Phiyz] = deal(ifftshift(Phixx, 1), ifftshift(Phiyy, 1), ifftshift(Phizz, 1), ifftshift(Phixy, 1), ifftshift(Phixz, 1), ifftshift(Phiyz, 1));
    [Phixx, Phiyy, Phizz, Phixy, Phixz, Phiyz] = deal(ifftshift(Phixx, 2), ifftshift(Phiyy, 2), ifftshift(Phizz, 2), ifftshift(Phixy, 2), ifftshift(Phixz, 2), ifftshift(Phiyz, 2));

    for i = 1:(N_z - 1)
        % Vectorial Beam Propagation Method
        [Ux, Uy, Uz] = vBPM(Ux, Uy, Uz, i);

        % Plot the field at n+1 slice
        updateFieldPlot(i + 1, hFig, hUx_abs, hUy_abs, hUz_abs, hUx_phase, hUy_phase, hUz_phase, Ux .* padMask, Uy .* padMask, Uz .* padMask);
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

    function [Ux_out, Uy_out, Uz_out] = vBPM(U_x, U_y, U_z, i)
        % Detialed in supplementary note 4

        % First, apply the free-space propagation kernel in k-space
        Ux_FT = fft2(U_x) .* BK .* naLPFilter;
        Uy_FT = fft2(U_y) .* BK .* naLPFilter;
        Uz_FT = fft2(U_z) .* BK .* naLPFilter;

        Ux_mid = ifft2(Ux_FT);
        Uy_mid = ifft2(Uy_FT);
        Uz_mid = ifft2(Uz_FT);

        % Second, apply the scattering potential in spatial domain as a phase modulation
        % Note that V = k0^2 * [n_medium^2 - n_particle^2]
        Ux_out = Ux_mid .* Phixx(:, :, i) + Uy_mid .* Phixy(:, :, i) + Uz_mid .* Phixz(:, :, i);
        Uy_out = Ux_mid .* Phixy(:, :, i) + Uy_mid .* Phiyy(:, :, i) + Uz_mid .* Phiyz(:, :, i);
        Uz_out = Ux_mid .* Phixz(:, :, i) + Uy_mid .* Phiyz(:, :, i) + Uz_mid .* Phizz(:, :, i);
    end

    function B_Kernal = vBPMKernal(d_z)
        KrShift = ifftshift(grid2_Kr);
        kzSquare = k_m ^ 2 - KrShift .^ 2;
        kzSquare(kzSquare < 0) = 0;
        Kz = sqrt(kzSquare);
        LPFilter = KrShift < (k_m);

        B_Kernal = LPFilter .* exp(-1i .* KrShift .^ 2 ./ (k_m + Kz) .* d_z);
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

        % Delay the input field with 0.86*pi for better phase visualization
        U_x = U_x * exp(1i * (-pi*0.86));
        U_y = U_y * exp(1i * (-pi*0.86));
        U_z = U_z * exp(1i * (-pi*0.86));

        % For Gaussian Beam
        % w0 = 5e-6;
        % U_x = U_x .* exp(-(grid2_X .^ 2 + grid2_Y .^2) / w0 ^ 2);
        % U_y = U_y .* exp(-(grid2_Y .^ 2 + grid2_Y .^2) / w0 ^ 2);
        % U_z = U_z .* exp(-(grid2_X .^ 2 + grid2_Y .^2) / w0 ^ 2);

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

        % Normalized k-space coordinates
        x = ax_Kx ./ k_m;
        y = ax_Ky ./ k_m;

        % Plot the PTFT Amplitude
        f_amp = figure('Name', 'PTFT Amplitude');
        tiledlayout(f_amp, 3, 3);
        nexttile;
        imagesc(x, y, abs(Q_xx'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{xx}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, abs(Q_xy'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{xy}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, abs(Q_xz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{xz}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);

        nexttile;
        imagesc(x, y, abs(Q_xy'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{yx}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        ylabel('k_y / k_m');
        nexttile;
        imagesc(x, y, abs(Q_yy'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{yy}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, abs(Q_yz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{yz}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);

        nexttile;
        imagesc(x, y, abs(Q_xz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{zx}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, abs(Q_yz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{zy}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        xlabel('k_x / k_m');
        nexttile;
        imagesc(x, y, abs(Q_zz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{zz}');
        clim([0, 1]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);

        cb = colorbar;
        cb.Layout.Tile = 'east';

        % Plot the PTFT Phase
        f_phase = figure('Name', 'PTFT Phase');
        tiledlayout(f_phase, 3, 3);
        nexttile;
        imagesc(x, y, angle(Q_xx'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{xx}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, angle(Q_xy'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{xy}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, angle(Q_xz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{xz}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);

        nexttile;
        imagesc(x, y, angle(Q_xy'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{yx}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        ylabel('k_y / k_m');
        nexttile;
        imagesc(x, y, angle(Q_yy'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{yy}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, angle(Q_yz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{yz}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);

        nexttile;
        imagesc(x, y, angle(Q_xz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{zx}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        nexttile;
        imagesc(x, y, angle(Q_yz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{zy}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        xlabel('k_x / k_m');
        nexttile;
        imagesc(x, y, angle(Q_zz'), 'AlphaData', ~clipping');
        axis xy image;
        title('Q_{zz}');
        clim([0, pi]);
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);

        cb = colorbar;
        cb.Layout.Tile = 'east';
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
        clim([0, 2.5]);
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
        clim([0, 2.5]);
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
        clim([0, 1.0]);
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
        n_beads_xx = 1.37; % Refractive Index of sphere particle
        n_beads_yy = 1.40;
        n_beads_zz = 1.44;
        particle_radius = 1.5e-6;
        particle_x = [-3, 3, -3, 3] * 1e-6;
        particle_y = [3, 3, -3, -3] * 1e-6;
        particle_z = [0, 0, 0, 0] * 1e-6;
        particle_rot_z = [0, 0, 0, 45];
        particle_rot_y = [90, 0, 0, 45];
        particle_rot_x = [0, 0, 90, 45];

        for i_bead = 1:numel(particle_x)
            % Euler rotation matrix
            R = eul2rotm([deg2rad(particle_rot_z(i_bead)), deg2rad(particle_rot_y(i_bead)), deg2rad(particle_rot_x(i_bead))]);
            epsilon = [n_beads_xx ^ 2, 0, 0; 0, n_beads_yy ^ 2, 0; 0, 0, n_beads_zz ^ 2];
            epsilon = R' * epsilon * R;
            delta_epsilon = [n_medium ^ 2, 0, 0; 0, n_medium ^ 2, 0; 0, 0, n_medium ^ 2] - epsilon;
            particle_Coord_R = (grid3_X - particle_x(i_bead)) .^ 2 + (grid3_Y - particle_y(i_bead)) .^ 2 + (grid3_Z - particle_z(i_bead)) .^ 2 < particle_radius ^ 2;
            V_xx(particle_Coord_R) = k0 .^ 2 * delta_epsilon(1, 1);
            V_yy(particle_Coord_R) = k0 .^ 2 * delta_epsilon(2, 2);
            V_zz(particle_Coord_R) = k0 .^ 2 * delta_epsilon(3, 3);
            V_xy(particle_Coord_R) = k0 .^ 2 * delta_epsilon(1, 2);
            V_xz(particle_Coord_R) = k0 .^ 2 * delta_epsilon(1, 3);
            V_yz(particle_Coord_R) = k0 .^ 2 * delta_epsilon(2, 3);
        end

        %% Use following code for anisotropic phantom in Fig.S3
        % n_beads_xx = 1.37; % Refractive Index of sphere particle
        % n_beads_yy = 1.44;
        % n_beads_zz = 1.44;
        % particle_radius = 1.5e-6;
        % epsilon = [n_beads_xx ^ 2, 0, 0; 0, n_beads_yy ^ 2, 0; 0, 0, n_beads_zz ^ 2];
        % epsilon0 = [n_medium ^ 2, 0, 0; 0, n_medium ^ 2, 0; 0, 0, n_medium ^ 2];
        % [grid3_Az, grid3_El, grid3_R] = cart2sph(grid3_X, grid3_Y, grid3_Z);
        % particle_Coord_R = find(grid3_R < particle_radius);
        % Az = grid3_Az(particle_Coord_R);
        % El = grid3_El(particle_Coord_R);
        % 
        % for i_pos = 1:numel(particle_Coord_R)
        %     R = eul2rotm([Az(i_pos), El(i_pos), 0]);
        %     delta_epsilon = epsilon0 - R' * epsilon * R;
        %     V_xx(particle_Coord_R(i_pos)) = k0 .^ 2 * delta_epsilon(1, 1);
        %     V_yy(particle_Coord_R(i_pos)) = k0 .^ 2 * delta_epsilon(2, 2);
        %     V_zz(particle_Coord_R(i_pos)) = k0 .^ 2 * delta_epsilon(3, 3);
        %     V_xy(particle_Coord_R(i_pos)) = k0 .^ 2 * delta_epsilon(1, 2);
        %     V_xz(particle_Coord_R(i_pos)) = k0 .^ 2 * delta_epsilon(1, 3);
        %     V_yz(particle_Coord_R(i_pos)) = k0 .^ 2 * delta_epsilon(2, 3);
        % end

    end

    function [Phi_xx, Phi_yy, Phi_zz, Phi_xy, Phi_xz, Phi_yz] = phaseModulation_beads(grid3_X, grid3_Y, grid3_Z)
        Phi_xx = complex(ones(size(grid3_X)));
        Phi_yy = complex(ones(size(grid3_X)));
        Phi_zz = complex(ones(size(grid3_X)));
        Phi_xy = complex(zeros(size(grid3_X)));
        Phi_xz = complex(zeros(size(grid3_X)));
        Phi_yz = complex(zeros(size(grid3_X)));

        % Fill the particle region with rotated scattering potential
        n_beads_xx = 1.37; % Refractive Index of sphere particle
        n_beads_yy = 1.40;
        n_beads_zz = 1.44;
        particle_radius = 1.5e-6;
        particle_x = [-3, 3, -3, 3] * 1e-6;
        particle_y = [3, 3, -3, -3] * 1e-6;
        particle_z = [0, 0, 0, 0] * 1e-6;
        particle_rot_z = [0, 0, 90, 45];
        particle_rot_y = [90, 0, 0, 45];
        particle_rot_x = [0, 0, 0, 45];

        for i_bead = 1:numel(particle_x)
            % Euler rotation matrix
            R = eul2rotm([deg2rad(particle_rot_z(i_bead)), deg2rad(particle_rot_y(i_bead)), deg2rad(particle_rot_x(i_bead))]);
            epsilon = [n_beads_xx ^ 2, 0, 0; 0, n_beads_yy ^ 2, 0; 0, 0, n_beads_zz ^ 2];
            epsilon = R' * epsilon * R;
            epsilon0 = [n_medium ^ 2, 0, 0; 0, n_medium ^ 2, 0; 0, 0, n_medium ^ 2];

            % According to Eq.(S42) in paper, the phase modulation matrix is a exponential matrix
            % The expm function is time consuming with Padé approximation algorithm behind
            % Don't use it in the pixelwise loop
            % The output matrix from expm of a symmetric matrix is also symmetric
            phi = expm(1i * k0 * d_z / 2 / n_medium * (epsilon - epsilon0));

            particle_Coord_R = (grid3_X - particle_x(i_bead)) .^ 2 + (grid3_Y - particle_y(i_bead)) .^ 2 + (grid3_Z - particle_z(i_bead)) .^ 2 < particle_radius ^ 2;
            Phi_xx(particle_Coord_R) = phi(1, 1);
            Phi_yy(particle_Coord_R) = phi(2, 2);
            Phi_zz(particle_Coord_R) = phi(3, 3);
            Phi_xy(particle_Coord_R) = phi(1, 2);
            Phi_xz(particle_Coord_R) = phi(1, 3);
            Phi_yz(particle_Coord_R) = phi(2, 3);
        end

        %% Use following code for anisotropic phantom in Fig.S3
        % n_beads_xx = 1.37; % Refractive Index of sphere particle
        % n_beads_yy = 1.44;
        % n_beads_zz = 1.44;
        % particle_radius = 1.5e-6;
        % epsilon = [n_beads_xx ^ 2, 0, 0; 0, n_beads_yy ^ 2, 0; 0, 0, n_beads_zz ^ 2];
        % epsilon0 = [n_medium ^ 2, 0, 0; 0, n_medium ^ 2, 0; 0, 0, n_medium ^ 2];
        % [grid3_Az, grid3_El, grid3_R] = cart2sph(grid3_X, grid3_Y, grid3_Z);
        % particle_Coord_R = find(grid3_R < particle_radius);
        % Az = grid3_Az(particle_Coord_R);
        % El = grid3_El(particle_Coord_R);
        % 
        % for i_pos = 1:numel(particle_Coord_R)
        %     R = eul2rotm([Az(i_pos), El(i_pos), 0]);
        %     phi = expm(1i * k0 * d_z / 2 / n_medium * (R' * epsilon * R - epsilon0));
        %     Phi_xx(particle_Coord_R(i_pos)) = phi(1, 1);
        %     Phi_yy(particle_Coord_R(i_pos)) = phi(2, 2);
        %     Phi_zz(particle_Coord_R(i_pos)) = phi(3, 3);
        %     Phi_xy(particle_Coord_R(i_pos)) = phi(1, 2);
        %     Phi_xz(particle_Coord_R(i_pos)) = phi(1, 3);
        %     Phi_yz(particle_Coord_R(i_pos)) = phi(2, 3);
        % end

    end

end
