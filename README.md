# BurgersEnergyTransfer
MATLAB code to perform optimization of initial condition exhibiting self-similar energy cascade for the one dimensional Burgers equation. 

# Files
- `Opt_01.m`: Script determines optimal initial condition exhibiting self-similar energy cascade for the 1D Burgers equation given values of $\nu$ and $\lambda$
- `BurgerOptFuncs.m`: Functions and parameters set required to run optimization and plotting of optimal solutions `Opt_01.m`

# How to Use
Set values of viscosity ($\nu$), time window ($T$), and parameter characterizing distance in Fourier space over which self-similar interactions occur ($\lambda$) in `Opt_01.m`. Numerical parameters and other optimization parameters can be adjusted in `BurgerOptFuncs.m`. Run optimization, and plots of optimal solutions will be generated once optimization is complete.

# Citing
Work has been published in Physical Review Fluids (PRFluids). The paper can be found [here](https://doi.org/10.1103/mxk3-jx9f).

Pritpal Matharu, Bartosz Protas, and Tsuyoshi Yoneda., (2026). *Unraveling self-similar energy transfer dynamics: A case study for the one-dimensional Burgers system.* Physical Review Fluids **11**, 034608, 2026, https://doi.org/10.1103/mxk3-jx9f

Bibtex:
```
@article{MatharuProtasYoneda_PhysRevFluids11_2026,
  title = {Unraveling self-similar energy transfer dynamics: A case study for the one-dimensional Burgers system},
  author = {Matharu, Pritpal and Protas, Bartosz and Yoneda, Tsuyoshi},
  journal = {Phys. Rev. Fluids},
  volume = {11},
  issue = {3},
  pages = {034608},
  numpages = {18},
  year = {2026},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/mxk3-jx9f},
  url = {https://link.aps.org/doi/10.1103/mxk3-jx9f}
}
```
