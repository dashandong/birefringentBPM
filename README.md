# birefringentBPM

A MATLAB program for simulating the scattering of birefringent objects using the Multi-slice Beam Propagation Method (BPM).

The program is based by the model from the paper "[Multi-slice computational model for birefringent scattering][paper]" by Shuqi Mu, Yingtong Shi, Yintong Song, Wei Liu, Wanxue Wei, Qihuang Gong, Dashan Dong, and Kebin Shi.

*If this code/program helps you, please cite our papers.*

## Features

- birefringentBPM.m: The program for simulating the scattering of birefringent objects. For figure 2 and figure S2, S4, & S5 in the paper.
- fullMueller.m: The program for generating the full Mueller matrix of the scattering system.

## Dependencies

- MATLAB R2019b or later with the following toolboxes:
  - [x] Navigation Toolbox (for eul2rotm function)

## Erratum

In the original paper, our model's simulation used the dielectric tensor exported from FDTD software to achieve better alignment with the coordinate system and coherent output. However, we mistakenly removed the off-diagonal elements in the dielectric tensor during the export process. This error only affects bead #4 in Figure 2 and Figures S2. You can obtain the correct results by using the birefringentBPM.m program available in this repository.

The model has been carefully re-verified with FTDT and COMSOL simulation, and there are no more errors in the analytical model or the simulation code. We apologize for any inconvenience this may have caused.

We thanks Dr. Gert-Jan Both for pointing out this error.

[paper]: https://doi.org/10.1364/OPTICA.472077