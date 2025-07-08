% BurgerOptFuncs.m: File containing functions for optimization of initial condition
% exhibiting self-similar energy cascade for the one dimensional Burgers equation given
% values of (nu) viscosity and (lambda) parametercharacterizing the Fourier distance which
% self-similar interactions occur.
%
% Author: Pritpal 'Pip' Matharu
% Applied Analysis Group
% Max Planck Institute for Mathematics in the Sciences
% Date: 2025/07/07
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loading file sets up the workspace, sets default values, and loads the functions
function BurgerOptFuncs
%% Initialize workspace
close all
% clc
set(groot, ...
    'defaulttextinterpreter', 'latex', ...
    'defaultLegendInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex', ...
    'defaultAxesFontSize',   20, ...
    'defaultTextFontSize',   20, ...
    'defaultTextFontSizeMode',   'manual', ...
    'defaultLegendFontSize', 20, ...
    'defaultLegendFontSizeMode', 'manual');
disp(datetime)
tic % Start timing function

%% Burgers Equation Parameters
% Numerical parameters
physP.Nval   = 1364;   % Number of dealiased points to use (340, 682, 1364, ...)
physP.dt     = 1e-3;   % Step size in time

%% Values for the IMEX method
% Coefficients from https://doi.org/10.1007/s10898-019-00855-1
% Journal of Global Optimization (2020) - Alimo, Cavaglieri, Beyhaghi, Bewley
alphacoeff = physP.dt *[343038331393.0/1130875731271.0 288176579239.0/1140253497719.0 ...
                        253330171251.0/677500478386.0  189462239225.0/1091147436423.0]';
betaIcoeff = physP.dt *[35965327958.0/140127563663.0   19632212512.0/2700543775099.0  ...
                      -173747147147.0/351772688865.0  91958533623.0/727726057489.0]';
betaEcoeff = physP.dt *[14.0/25.0                      777974228744.0/1346157007247.0 ...
                        251277807242.0/1103637129625.0 113091689455.0/220187950967.0]';
gammacoeff = physP.dt *[0.0                           -251352885992.0/790610919619.0 ...
                       -383714262797.0/1103637129625.0 -403360439203.0/1888264787188.0]';

%% Initialize Parameters for Optimization
% Problem Setup
optP.H1grad = 1;   % Type of Sobolev gradient (H1 norm)
optP.H1     = 2;   % Constraint (H1 semi-norm)
optP.l1     = 10;  % Sobolev length scale parameter, ensures suitable regularity
% Adjustable optimization parameters
optP.tol        = 1e-8; % Tolerance for optimization
optP.max_iter   = 100;  % Maximum number of iterations
optP.fq         = 10;   % Assign value of clearing frequency of PR beta value
optP.tau0       = 1.0;  % Starting step length for line search
optP.brack_fact = 2;    % Bracketing Factor, to enlarge & shrink the fminbnd interval

%% Assign function calls and parameters in file
assignin('caller', 'physP',             physP); % Physical parameters
assignin('caller', 'optP',              optP);  % Optimization parameters

assignin('caller', 'burgers_init',      @burgers_init);
assignin('caller', 'BurgersSolve',      @BurgersSolve);
assignin('caller', 'AdjointSolve',      @AdjointSolve);
assignin('caller', 'Transform2Fourier', @Transform2Fourier);
assignin('caller', 'Transform2Real',    @Transform2Real);
assignin('caller', 'ProjConstr',        @ProjConstr);
assignin('caller', 'Sobolevgrad',       @Sobolevgrad);
assignin('caller', 'innerprod_H1',      @innerprod_H1);
assignin('caller', 'costfunF',          @costfunF);
assignin('caller', 'costfunfwd',        @costfunfwd);
assignin('caller', 'costfunmin',        @costfunmin);
assignin('caller', 'pltFuncs',          @pltFuncs);

% Placeholder for static variables to be declared
svars.N = [];    
assignin('caller', 'svars',             svars);  % Static variables

%% *********************************** FUNCTIONS *****************************************
% ----------------------------------------------------------------------------------------
% FUNCTION: burgers_init
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Initialize the Burgers Equation solver and optimization
%
% INPUT
% physIn ....... Physical parameters for problem setup (lambda, nu, T)
% ( physP ) .... Uses numerical parameters for solving Burgers equation
%
% OUTPUT
% uIC .......... Initial guess for initial condition, discretized equispaced
% str .......... String for saving results
% svarsO.x ......... Spatial domain, discretized equispaced (0 <= x <= 2pi)
% svarsO.lk ........ Indices for shifting wavenumbers, based on lambda
% svarsO.kvals ..... Wavenumbers to be used in Fourier space
% svarsO.dealias ... Filtering for dealiasing wavenumbers
% svarsO.dx ........ Step size in spatial domain
% svarsO.N ......... Number of points given by 2/3 rule
% svarsO.N2 ........ Number of points in extended grid, based on lambda
% svarsO.Nt ........ Reference length for number of time steps
%
% FORMAT
% [uIC, str, svarsO] = burgers_init(physP)
%
% -------------------------------------------------------------------------
    function [uIC, str, svarsO] = burgers_init(physIn)

        %% Set physical parameters so they are available to all functions
        physP.lambda = physIn.lambda;
        physP.nu     = physIn.nu;
        physP.T      = physIn. T;

        %% Set Numerical values
        % Determine extended grid points, based on lambda value
        Nmax  = physP.lambda*physP.Nval; svars.N2tmp = 3 * Nmax / 2;
        % Discretization points in extended grid
        svars.N2    = 2.^(ceil(log2(svars.N2tmp)));

        % Number of time steps
        svars.Nt = round(physP.T/physP.dt);
        % Two-thirds rule for determining number of points
        svars.N = round(2 * svars.N2 / 3);
        if (mod(svars.N,2) ~= 0); svars.N = svars.N - 1; end % FFT require even points

        % Indices for shifting wavenumbers
        svars.lk = (1:physP.lambda:svars.N/2)';

        % Wavenumbers
        k                        = zeros(svars.N2, 1);
        k(1:svars.N2/2)          = 0:svars.N2/2-1;       % Zeroth and positive wavenumbers
        k(svars.N2/2+1:svars.N2) = (svars.N2/2:svars.N2-1)-svars.N2; % Negative wavenumbers
        % Extract relevant wavenumbers
        svars.kvals               = zeros(svars.N2/2, 1);
        svars.kvals(1:svars.N2/2) = k(1:svars.N2/2);

        % Use a Gaussian spectral filter for dealiasing
        svars.dealias = exp( -36 * (k(1:svars.N2/2)/svars.N).^36 );

        %% Initial Condition
        % Generating the initial condition (a simple sine wave)
        svars.dx  = 2.0 * pi / svars.N2;
        svars.x   = (0.0:svars.dx:(svars.N2-1)*svars.dx)';
        uIC       = sin( -svars.x );

        % Check value of norm of initial condition for the given constraint ( =pi )
        svars.cval = innerprod_H1(uIC, uIC, optP.H1);
        % Ensure initial guess satisfies constraint value
        uIC = ProjConstr(uIC);

        %% Strings for saving
        % Sobolev parameter
        l1coeff = floor(log10(optP.l1));
        l1fac   = floor(optP.l1/(10^l1coeff));
        % Viscosity
        nucoeff = floor(log10(physP.nu))-1;
        nufac   = floor(physP.nu/(10^(nucoeff)));
        % Time
        Tcoeff = floor(log10(physP.T))-1;
        Tfac   = round(physP.T/(10^Tcoeff));
        % Time step
        dtcoeff = floor(log10(physP.dt));
        dtfac   = floor(physP.dt/(10^dtcoeff));
        % Saving variables
        str  = sprintf('Burg1D_Optimize_N%d_dt%de%d_L%de%d_T%de%d_lambda%d_nu%de%d.mat', ...
            svars.N2, dtfac, dtcoeff, l1fac, l1coeff, Tfac, Tcoeff, physP.lambda, nufac, nucoeff);

        % Static variables for problem step-up
        svarsO = svars;
    end % End of function burgers_init

% ----------------------------------------------------------------------------------------
% FUNCTION: BurgersSolve
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Solves the Burgers equation via a DNS using a pseudo-spectral Galerkin approach 
% with dealiasing and a globally third-order, four step IMEX time stepping method 
% (which maintains good stability)
%
% INPUT
% u0 ........... Initial Condition Function
% 
% USES
% svars ........ Static variables defined by burgers_init
% physP.nu ..... Viscosity Parameter
%
% OUTPUT
% usave ........ Solution in real space at all points in space and time
% uhsave ....... Solution in real space at all points in space and time
%
% FORMAT
% [usave, uhsave] = BurgersSolve( u0 )
%
% ----------------------------------------------------------------------------------------
    function [usave, uhsave] = BurgersSolve( u0 )

        % Empty storage arrays
        usave  = zeros(svars.N2,  svars.Nt+1);
        uhsave = zeros(svars.N/2, svars.Nt+1);

        % Save initial condition
        u            = u0;
        usave(:, 1)  = u;
        % Transform to Fourier space
        u_hat        = Transform2Fourier( u );
        uhsave(:, 1) = u_hat(1:svars.N/2);

        % Assembling multiplication vectors;
        Kx(1:svars.N2/2+1, 1)   = ((0:svars.N2/2) * 1i);
        Kx2(1:svars.N2/2+1, 1)  = - ((0:svars.N2/2)  .* (0:svars.N2/2)) * physP.nu;

        % To be used in the IMEX method
        uu_hat0(1:svars.N/2, 1) = 0.0;

        %% Main loop
        for k=2:svars.Nt+1
            % IMEX time stepping
            for rk=1:4
                % Pseudospectral calculation of the nonlinear products
                u                 = Transform2Real( u_hat );
                % Calculate nonlinear term in real space
                uu(1:svars.N2, 1) =  0.5 * (u(1:svars.N2) .* u(1:svars.N2));
                % Transform back to Fourier space
                uu_hat            = Transform2Fourier( uu );

                % The explicit part
                uu_hat(1:svars.N/2)   = - Kx(1:svars.N/2) .* uu_hat(1:svars.N/2);
                % Compute solution at next sub-step using IMEX time-stepping method
                uhat2(1:svars.N/2, 1) = ( ...
                    ( 1.0 +  betaIcoeff(rk) * Kx2(1:svars.N/2)) .* u_hat(1:svars.N/2) + ...
                             betaEcoeff(rk) * uu_hat(1:svars.N/2) + ...
                             gammacoeff(rk) * uu_hat0(1:svars.N/2) ...
                    )./ ...
                    ( 1.0 -  alphacoeff(rk) * Kx2(1:svars.N/2) );


                % Solution in Fourier space
                u_hat(1:svars.N/2)   = uhat2(1:svars.N/2);
                % Update explicit part, for next substep
                uu_hat0(1:svars.N/2) = uu_hat(1:svars.N/2);

            end % End of RK steps

            %% Output data
            u = Transform2Real( u_hat );
            usave(1:svars.N2, k)    = u(1:svars.N2);
            uhsave(1:svars.N/2, k)  = u_hat(1:svars.N/2);

            % Setting up nonlinear vectors for the first substep
            uu_hat0(1:svars.N/2, 1) = 0.0;

        end % End of time stepping

    end % End of BurgersSolve

% ----------------------------------------------------------------------------------------
% FUNCTION: AdjointSolve
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Solves the Adjoint System and for determining the L2 gradient
%
% INPUT
% usave ........ Solution in real space at all points in space and time
% RHS .......... Terminal condition in Fourier space for adjoint
%
% USES
% svars ........ Static variables defined by burgers_init
% physP.nu ..... Viscosity Parameter
%
% OUTPUT
% z ............ Solution of Adjoint system at t = 0
%
% FORMAT
% z = AdjointSolve( u_save, RHS )
%
% ----------------------------------------------------------------------------------------
    function z = AdjointSolve( u_save, RHS )

        % Set initial (terminal) condition
        z_hat =  RHS*(svars.N2/2);

        % Assembling multiplication vectors;
        Kx(1:svars.N2/2+1, 1)   = ((0:svars.N2/2) * 1i);
        Kx2(1:svars.N2/2+1, 1)  = - ((0:svars.N2/2)  .* (0:svars.N2/2)) * physP.nu;

        %% Main loop
        for k = svars.Nt:-1:1
            % Setting up nonlinear vectors for the first substep
            nonlin_hat0(1:svars.N/2, 1) = 0.0;
            % Obtain velocity solution
            u     = u_save(:, k+1);
            % Transform velocity to Fourier space
            u_hat = Transform2Fourier( u );

            % Transform to real space
            u  = Transform2Real( u_hat );

            % IMEX time stepping
            for rk=1:4
                %% Pseudospectral calculation of the nonlinear products
                % Transform to real space
                dz = Transform2Real( Kx(1:svars.N/2).*z_hat(1:svars.N/2) );
                % Calculate nonlinear terms in real space
                udz(1:svars.N2, 1) =  u(1:svars.N2) .* dz(1:svars.N2);
                % Transform to Fourier space
                udz_hat            = Transform2Fourier( udz );
                % The explicit part
                nonlin_hat(1:svars.N/2, 1) = udz_hat(1:svars.N/2);

                % Compute solution at next sub-step using IMEX time-stepping method
                zhat2(1:svars.N/2, 1) = ( ...
                    ( 1.0 +  betaIcoeff(rk) * Kx2(1:svars.N/2)) .* z_hat(1:svars.N/2) + ...
                             betaEcoeff(rk) * nonlin_hat(1:svars.N/2) + ...
                             gammacoeff(rk) * nonlin_hat0(1:svars.N/2) ...
                    )./ ...
                    ( 1.0 -  alphacoeff(rk) * Kx2(1:svars.N/2) );

                % Solution in Fourier space
                z_hat(1:svars.N/2)       = zhat2(1:svars.N/2);
                % Update explicit part, for next substep
                nonlin_hat0(1:svars.N/2) = nonlin_hat(1:svars.N/2);

            end % End of RK steps
        end % End of time stepping
        %% Output data
        z = Transform2Real( z_hat );

end % End of AdjointSolve

% ----------------------------------------------------------------------------------------
% FUNCTION: costfunF
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% The function determines the cost functional value 
%
% INPUT
% uIC ........ Initial condition
% uEnd ....... Solution at final time
%
% USES
% svars ........... Static variables defined by burgers_init
% physP.lambda .... Parameter in cost functional
%
% OUTPUT
% J ......... Value of cost functional
% Jadj ...... Expression used in adjoint equation
% Jsimpl .... Expression used in simple Riesz form
% Jint ...... Summand of Functional
%
% FORMAT
% [J, Jadj, Jsimpl] = costfunF( uIC, uEnd )
%
% ----------------------------------------------------------------------------------------
    function [J, Jadj, Jsimpl, Jint] = costfunF( uIC, uEnd )

        % Storage vectors
        uf1  = zeros(svars.N2/2, 1);
        uf2  = zeros(svars.N2/2, 1);

        % Transform to Fourier space
        uhIC  = Transform2Fourier(uIC)/(svars.N2/2);
        uh    = Transform2Fourier(uEnd)/(svars.N2/2);

        % Terms scaled with lambda
        % Wavenumbers scaled
        uf2(svars.lk, 1)  = uh(svars.lk);

        % Initial condition, wavenumbers shifted
        uf1(svars.lk)     = uhIC(1:length(svars.lk));

        % Summand of functional
        Jint    = abs(uf1).^2 - physP.lambda*(abs(uf2).^2);
        Jint(1) = 0.0; % Safety for zero mean solutions

        % Cost functional
        J    = 0.5*sum(abs(Jint).^2);
        % RHS for adjoint equation
        Jadj = -(physP.lambda/pi)*Jint.*uf2;

        % For computing the simple Riesz expression
        % Scaling including lambda
        uf3                        = zeros(svars.N2/2, 1);
        uf3(1:length(svars.lk), 1) = uh(svars.lk);
        % Term for simple Riesz expression
        Jsimpl    = abs(uhIC).^2 - physP.lambda*(abs(uf3).^2);
        Jsimpl(1) = 0.0;

    end % End of costfunF

% ----------------------------------------------------------------------------------------
% FUNCTION: Transform2Fourier
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Wrapper for real to Fourier space transforms, using conjugate symmetry and dealiasing.
%
% INPUT
% ur ......... Function in real space
%
% USES
% svars ...... Static variables defined by burgers_init
%
% OUTPUT
% uf ......... Fourier transform of function
%
% FORMAT
% uf = Transform2Fourier( ur )
%
% ----------------------------------------------------------------------------------------
    function uf = Transform2Fourier( ur )

        % Perform the fft on the extended grid
        utmp = fft(ur, svars.N2);

        % Gaussian dealiasing and only half data due to real valued FT
        uf   = utmp(1:svars.N2/2).*svars.dealias;

    end % End of Transform2Fourier

% ----------------------------------------------------------------------------------------
% FUNCTION: Transform2Real
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Wrapper for Fourier to real space transforms, using conjugate symmetry and padding.
%
% INPUT
% uf ......... Fourier transform of function
%
% USES
% svars ...... Static variables defined by burgers_init
%
% OUTPUT
% ur ......... Function in real space
%
% FORMAT
% ur = Transform2Real( uf )
%
% ----------------------------------------------------------------------------------------
    function ur = Transform2Real( uf )

        % Create vector to transform into real with added oddball
        utmp                              = zeros(svars.N2, 1);
        % For filtering, only take upto maximum resolved wavenumber
        utmp(1:svars.N/2, 1)              = uf(1:svars.N/2);
        % Using conjugate symmetry, fill in the negative wavenumber entries
        utmp(svars.N2:-1:svars.N2/2+2, 1) = conj(utmp(2:svars.N2/2));

        % Transform into real space
        ur(1:svars.N2, 1) = real(ifft(utmp,svars.N2));

    end % End of Transform2Real

% ----------------------------------------------------------------------------------------
% FUNCTION: ProjConstr
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Projects the function onto the constraint and ensure mean zero
%
% INPUT
% f ......... Function, not on constraint manifold
%
% USES
% svars ...... Static variables defined by burgers_init
% optP.H1 .... Constraint norm flag for problem 
%
% OUTPUT
% g .......... Function  on constraint manifold with zero mean
%
% FORMAT
% g = ProjConstr( f )
%
% ----------------------------------------------------------------------------------------
    function g = ProjConstr(f)

        % Compute inner product, and normalize to satisfy constraint
        g = (sqrt(svars.cval)*f)/sqrt(innerprod_H1(f, f, optP.H1));

        % Ensure zeroth wavenumber is zero
        gh    = fft(g, svars.N2);
        gh(1) = 0.0;
        g     = real(ifft(gh)); % Return back to real space

    end % End of ProjConstr

% ----------------------------------------------------------------------------------------
% FUNCTION: Sobolevgrad
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Determine the Sobolev gradient, defined by H1 norm
%
% INPUT
% grad_L2 ......... L2 gradient
%
% USES
% svars.kvals ..... Static variable, wavenumbers
% optP.H1 ......... Constraint norm flag for problem 
%
% OUTPUT
% grad_H .......... Sobolev gradient
%
% FORMAT
% grad_H = Sobolevgrad( grad_L2 )
%
% ----------------------------------------------------------------------------------------
    function grad_H = Sobolevgrad( grad_L2 )

        % Transform to Fourier space
        grad_hat = Transform2Fourier(grad_L2);

        % Compute sobolev gradient (filter due to periodic BCs)
        grad_hat = grad_hat./(1 + (optP.l1^2)*(svars.kvals).^2);

        % Ensure mean zero
        grad_hat(1) = 0.0;

        % Transform back to physical space
        grad_H   = Transform2Real (grad_hat);

    end % End of Sobolevgrad

% ----------------------------------------------------------------------------------------
% FUNCTION: innerprod_H1
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Determine inner product, based on specified space
%
% INPUT
% f ............... Function 1
% g ............... Function 2
% H1 .............. Flag for inner product space
%
% USES
% svars.kvals ..... Static variable, wavenumbers
% optP.l1 ......... Sobolev parameter
% optP.H1 ......... Constraint norm flag for problem 
%
% OUTPUT
% inval ........... Value of resulting inner product
%
% FORMAT
% inval = innerprod_H1( f, g, H1 )
%
% ----------------------------------------------------------------------------------------
    function inval = innerprod_H1( f, g, H1 )

        % Determine whether to compute H1 or L2 innerproduct
        if (H1 == 1) % H1 norm
            f_hat = Transform2Fourier(f);
            % Periodic domain, so we can integrate by parts and compute by one minus
            % Laplacian on one function
            ftmp_hat = (1 + (optP.l1^2)*(svars.kvals).^2).*f_hat;
            % Transform back to physical space
            ftmp   = Transform2Real (ftmp_hat);
        elseif (H1 == 2) % H1 semi-norm
            f_hat = Transform2Fourier(f);
            % Periodic domain, so we can integrate by parts and compute by one minus 
            % Laplacian on one function
            ftmp_hat = ((svars.kvals).^2).*f_hat;
            % Transform back to physical space
            ftmp   = Transform2Real (ftmp_hat);
        else % L2 inner product
            ftmp = f;
        end

        % Compute L2 inner product
        inval = trapz(svars.x, ftmp.*g);

    end % End of innerprod_H1

% ----------------------------------------------------------------------------------------
% FUNCTION: costfunF
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Computes the cost functional by first solving burgers equation
%
% INPUT
% uIC ........ Initial condition
%
% OUTPUT
% J ......... Value of cost functional
%
% FORMAT
% J = costfunfwd( uIC )
%
% ----------------------------------------------------------------------------------------
    function J = costfunfwd(uIC)

        % Ensure norm is equal to cval
        uIC = ProjConstr(uIC);

        % Solve Burgers equation
        [u, ~] = BurgersSolve( uIC );

        % Calculate Cost Functional
        J   = costfunF(uIC, u(:, end));

    end % End of costfunfwd

% ----------------------------------------------------------------------------------------
% FUNCTION: costfunmin
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Minimizing the cost functional by seeking the optimal step length to take 
%
% INPUT
% uIC ........ Initial condition
% Jgrad ...... Gradient of cost functional
% tau ........ Initial step length
% history .... History, containing step length, and value of cost functional
%
% OUTPUT
% tau ........ Optimal step length
% fval ....... Value of cost functional, corresponding to optimal tau
% history .... History, containing step length, and value of cost functional
% exitflag ... Exitflag, to ensure the minimization performed correctly
% output ..... Diagnostic, of the minimization process
%
% FORMAT
% [tau, fval, history, exitflag, output] = costfunmin( uIC, Jgrad, tau, history )
%
% ----------------------------------------------------------------------------------------
    function [tau, fval, history, exitflag, output] = ...
            costfunmin( uIC, Jgrad, tau, history )

        % Set options, to output the history
        options = optimset('OutputFcn', @myoutput);

        % The cost functional is line-minimized with respect to 's'
        [tau, fval, exitflag, output] = fminbnd( @(s) ...
            costfunfwd(uIC + s*Jgrad),...
            0.0, tau, options);

        % Nested function, to output the history
        function stop = myoutput(x, optimvalues, state)
            % Flag for exiting function
            stop = false;
            % If still iterating, update history vector
            if isequal(state, 'iter')
                % Append the step-size and corresponding value to history
                history = [history; [x, optimvalues.fval] ];
            end
        end
    end % End of costfunmin

% ----------------------------------------------------------------------------------------
% FUNCTION: pltFuncs
%
% AUTHOR ... Pritpal Matharu
% DATE ..... 2025/07/07
%
% Minimizing the cost functional by seeking the optimal step length to take 
%
% INPUT
% u0 ......... Optimal initial condition
% u .......... Corresponding solution
% J .......... Iterations of cost functional
%
%
% FORMAT
% pltFuncs( u0, u, J )
%
% ----------------------------------------------------------------------------------------
    function pltFuncs( u0, u, J )

        % Setup plotting variables
        line_sty = {'-', '--', '-.k', '--', '--', '--'};
        LegendT{1} = sprintf('$t=0$');
        LegendT{2} = sprintf('$t=T$');

        % Determine summand of cost functional
        [~, ~, ~, J_int] = costfunF( u0, u(:, end) );

        %% Determine the space-time averaged energy dissipation
        t                     = (0:physP.dt:physP.T)';   % Time vector
        Kx(1:svars.N2/2, 1)   = ((0:svars.N2/2-1) * 1i); % Derivative operator
        Enst                  = zeros(svars.Nt+1, 1);    % Enstrophy
        du                    = zeros(size(u));          % Derivative of solution
        % Loop through each time step and determine derivative and enstrophy
        for jj = 1:length(Enst)
            u_hat     = Transform2Fourier( u(:, jj) ); % Fourier transform
            dutmp     = Transform2Real( Kx .* u_hat ); % Derivative, return to real space
            du(:, jj) = dutmp;                         % Store derivative term
            Enst(jj)  = trapz(svars.x, dutmp.*dutmp);  % Enstrophy
        end

        % Determine the time averaged mean enstrophy
        Enst_mean = trapz(t, Enst/pi)/physP.T;
        % Determine eddy turnover time
        te        = (Enst_mean^(-1/2));
        fprintf('\n Constant C_T = %d \n \n ', physP.T/te);

        %% Plot Cost functional iterations
        figure(); clf;
        axCost = gca;
        hold on
        box on
        axCost.ColorOrderIndex = 1;
        plot(0:length(J)-1, J/J(1), line_sty{1})
        xlabel('$n$')
        ylabel('$\mathcal{J}_{\nu, \lambda}(\phi^{(n)})$')
        xlim([0, length(J)])
        set(axCost, 'yscale', 'log')
        strT = sprintf('Normalized Cost Functional');
        title(strT)

        %% Cost Functional Summand
        figure(); clf;
        ax_hatint = gca;
        hold on
        box on
        ax_hatint.ColorOrderIndex = 4;
        plot(1:svars.N2/2-1,abs(J_int(2:end)), '.')
        set(ax_hatint,'xscale','log')
        set(ax_hatint,'yscale','log')
        xlabel('$k$')
        ylabel('$f(k;\varphi_{\nu, \lambda})$')
        strT = sprintf('Cost Functional Summand');
        title(strT)

        %% Physical space
        figure()
        ax_real = gca;
        hold on
        box on
        plot(svars.x, u(:, 1),   line_sty{1},   'Color', [0 0.4470 0.7410])
        plot(svars.x, u(:, end), line_sty{end}, 'Color', [0.8500 0.3250 0.0980])
        xlim([0 2*pi])
        xlabel('$x$')
        ylabel('$u(t, x; \varphi_{\nu, \lambda})$')
        set(ax_real,'XTick',0:pi/2:2*pi)
        set(ax_real,'XTickLabel',{'0','$\frac{\pi}{2}$','$\pi$','$\frac{3\pi}{2}$','$2\pi$'})
        strT = sprintf('Optimal Physical Space Solution');
        title(strT)
        legend(LegendT, 'location', 'best')

        %% Final time with gradients
        figure(); clf;
        ax_duF = gca;
        hold on
        box on
        yyaxis right
        ax_duF.YColor = [0 0 0];
        plot(svars.x, abs(du(:, end)), '-', 'Color', [0 0 0])
        plot(svars.x, abs(du(:, end)), '.', 'Color', [0 0 0])
        set(ax_duF, 'yscale', 'log')
        ylim([1e-2 1e6])
        ylabel('$|\frac{\partial u(T, x; \varphi_{\nu, \lambda})}{\partial x}|$')
        yyaxis left
        ax_duF.YColor = [0.8500 0.3250 0.0980];
        ax_duF.ColorOrderIndex = 2;
        plot(svars.x, u(:, end), '--', 'LineWidth', 1, 'Color', [0.8500 0.3250 0.0980])
        ylabel('$u(T, x; \varphi_{\nu, \lambda})$')
        xlim([0 2*pi])
        xlabel('$x$')
        set(ax_duF,'XTick',0:pi/2:2*pi)
        set(ax_duF,'XTickLabel',{'0','$\frac{\pi}{2}$','$\pi$','$\frac{3\pi}{2}$','$2\pi$'})
        strT = sprintf('Final Time Solution');
        title(strT, 'Interpreter', 'Latex')

        %% Fourier Space
        figure()
        ax_hat = gca;
        hold on
        box on
        % Time 0
        uhat = fft(u(:, 1),svars.N2)/(svars.N2/2);
        uhat = uhat(1:svars.N/2);
        ax_hat.ColorOrderIndex = 1;
        loglog(1:svars.N/2-1,abs(uhat(2:end)), line_sty{1})
        % Final time
        uhat = fft(u(:, end),svars.N2)/(svars.N2/2);
        uhat = uhat(1:svars.N/2);
        ax_hat.ColorOrderIndex = 2;
        loglog(1:svars.N/2-1,abs(uhat(2:end)), line_sty{end})
        set(ax_hat,'yscale','log')
        set(ax_hat,'xscale','log')
        xlim([1 1e3])
        ylim([1e-15 1])
        xlabel('$k$')
        ylabel('$|\hat{u}(t, k; \varphi_{\nu, \lambda})|$')
        strT = sprintf('Fourier Space Solution');
        title(strT)
        legend(LegendT, 'location', 'northeast')

        %% Enstrophy
        figure()
        ax_enst = gca;
        hold on
        box on
        ax_enst.ColorOrderIndex = 4;
        plot(t, Enst(:, 1), '-')
        xlabel('$t$')
        ylabel('$\mathcal{E}(u(t, \cdot))$')
        xlim([0 physP.T])
        strT = sprintf('Enstrophy'); 
        title(strT)

    end % End of pltFuncs

end % End of Burgers BurgerOptFuncs