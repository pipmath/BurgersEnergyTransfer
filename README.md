# BurgersEnergyTransfer
MATLAB code to perform optimization of initial condition exhibiting self-similar energy cascade for the one dimensional Burgers equation. 

# Files
- `Opt_01.m`: Script determines optimal initial condition exhibiting self-similar energy cascade for the 1D Burgers equation given values of $\nu$ and $\lambda$
- `BurgerOptFuncs.m`: Functions and parameters set required to run optimization and plotting of optimal solutions `Opt_01.m`

# How to Use
Set values of viscosity ($\nu$), time window ($T$), and parameter characterizing distance in Fourier space over which self-similar interactions occur ($\lambda$) in `Opt_01.m`. Numerical parameters and other optimization parameters can be adjusted in `BurgerOptFuncs.m`. Run optimization, and plots of optimal solutions will be generated once optimization is complete.
