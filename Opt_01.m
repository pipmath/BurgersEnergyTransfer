% Burgers_Optimization.m: Script determines the optimal initial condition exhibiting 
% self-similar energy cascade for the one dimensional Burgers equation given values of 
% (nu) viscosity and (lambda) parametercharacterizing the Fourier distance which 
% self-similar interactions occur. 
%
% Author: Pritpal 'Pip' Matharu
% Applied Analysis Group
% Max Planck Institute for Mathematics in the Sciences
% Date: 2025/07/07
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
%% Initialize solvers, functions, and parameters 
% Initialize workspace and load functions
BurgerOptFuncs;
% Physical Parameters
physP.lambda = 2;      % Lambda value in functional
physP.nu     = 2.5e-3; % Viscosity
physP.T      = 0.6;    % Final time
% Setup problem and PDE variables
[uIC, str, svars] = burgers_init( physP );
tic
disp('============================================================================');
disp(datetime);
fprintf('\n Constraint value: %d \n', svars.cval)
fprintf(' Computing IC for: nu = %d,  lambda = %d \n', ...
                      physP.nu, physP.lambda)
fprintf(' Parameters: N2 = %d,  T = %d,  dt = %d, l1 = %d \n \n', ...
                svars.N2, physP.T, physP.dt, optP.l1)
disp('============================================================================');

%% Calculate values for initial guess
% Solve Burgers equation
[u, uhat] = BurgersSolve( uIC );

% Calculate Cost Functional, and output terms to compute gradient
[J1, Jadj_hat, Jsimpl_hat] = costfunF( uIC, u(:, end) );
J2 = J1*1e10; % Dummy value
fprintf('Initial Cost Functional: %d \n', J1);

%% Storage arrays for iterations
p     = 1;                                   % Initialization of iteration counter
IC_1  = zeros(length(uIC), optP.max_iter+1); % Initial conditions
J     = zeros(optP.max_iter+1, 1);           % Cost Functional
gradS = zeros(length(uIC), optP.max_iter);   % Sobolev gradient applied
CG    = zeros(length(uIC), optP.max_iter);   % Conjugate gradients
Tau   = zeros(optP.max_iter+1, 1);           % Step lengths for applying gradient
Exit  = zeros(optP.max_iter, 1);             % Exit flags from line search method
Num   = zeros(optP.max_iter, 1);             % Number of function evaluations
bval  = zeros(optP.max_iter, 1);             % Momentum term from Polak-Ribiere method

% Store values for initial guess
J(p, 1)    = J1;        % Cost Functional
IC_1(:, p) = uIC;       % Initial conditions
Tau(p, 1)  = optP.tau0; % Step lengths for applying gradient
u0         = uIC;       % Current initial condition

%% Loop Iteration
J3 = 0.0; % Dummy value

% Iterate until condition met, maximum iteration is reached, or functional too small
while ( (abs(J1 - J2))/(abs(J1)) > optP.tol && p <= optP.max_iter && abs(J2) > 1e-12 )

    %% Determine Sobolev Gradient
    % Compute adjoint solution (backwards in time) for gradient term
    z       = AdjointSolve( u, Jadj_hat );
    % Simple Riesz term (accountinig for numerical scalings in Fourier space)
    u0_hat  = Transform2Fourier( u0 )/(svars.N2/2);         % Fourier transform IC
    t1h     = Jsimpl_hat(1:svars.N/2).*u0_hat(1:svars.N/2); % Compute product
    t1      = Transform2Real( t1h*(svars.N2/2) );           % Return to real space
    % Combine to compute L2 gradient
    grad_L2 = 2*(t1/pi + z);

    % Determine H1 Sobolev gradient, for additional regularity
    % Take negative value, to apply steepest DESCENT direction
    del_J   = -Sobolevgrad( grad_L2 );
    
    %% Conjugate Gradient method
    % CG with the Polak-Ribiere method including frequency resetting
    if p >= 2 && mod(p, optP.fq)~=0
        % Using the Polak Ribere    
        bnumer = innerprod_H1( del_J, del_J - delk, optP.H1grad );
        bdenom = innerprod_H1( delk, delk, optP.H1grad );
        bPR    = bnumer/bdenom;
        
        % Use value to create the conjugate gradient
        PR  = del_J + bPR.*pr;
    else
        % Frequency clearing to reset the conjugate-gradient procedure
        bPR = 0; % Save for diagnostics
        
        % Use standard gradient (previous gradient cleared)
        PR  = del_J;
    end % End of conjugate gradient statement

    % Ensure that the cost funcction does not equal zero (only for the iteration = 1)
    if p ~= 1
        % Set Cost functional equal to current iteration
        J1 = J2;
    end
        
    % Evaluate cost functional
    J2 = costfunfwd( u0+ optP.tau0*PR );
    
    % Bracketing the minimum from the right, for the Brent method
    Feval = 1;              % Count number of function calls
    Gtau  = [optP.tau0 J2]; % Store step length and respective value of cost function

    %% Bracket interval for step length
    if J2 > J1
        % If the Cost Functional at tau0 is GREATER than the current iteration of
        % the evaluated Cost Functional, we try to shrink the bracket interval that
        % we evaluate the fminbnd function
        
        % Shrink bracket interval, until appropriate value is found
        while J2 > J1 && optP.tau0 > 1e-8
            % Shrink by a constant bracketing factor
            optP.tau0 = optP.tau0/optP.brack_fact;
            % Calculate the cost function for the given step length and gradient
            J2        = costfunfwd( u0+ optP.tau0*PR );
            % Update the number of times the function is called
            Feval     = Feval + 1;
            % Update store the step length used, and the value of cost function
            Gtau      = [Gtau; [optP.tau0, J2] ];
        end
        % To ensure that we did not shrink the bracket too much!
        optP.tau0 = optP.tau0*optP.brack_fact;
        
    else
        % If the Cost Functional at tau0 is less than the current iteration of the
        % evaluated Cost Functional, we try to expand the bracket that we evaluate
        % the fminbnd function at because there may exist an even larger interval
        % that could give us an even smaller (and better) step length for the Cost
        % Functional!
        
        % Expand bracket interval, until appropriate value is found
        while J2 < J1 && (abs(J3 - J2)/abs(J2) > optP.tol)
            % Expand by a constant bracketing factor
            optP.tau0 = optP.tau0*optP.brack_fact;            
            % Store for check criterion
            J3        = J2;
            % Calculate the cost function for the given step length and gradient
            J2        = costfunfwd( u0+ optP.tau0*PR );
            % Update the number of times the function is called
            Feval     = Feval + 1;
            % Update store the step length used, and the value of cost function
            Gtau      = [Gtau; [optP.tau0, J2] ];
        end
    end % End of if statement
    
    %% Determine optimal value for steplength
    % Determine step length along the gradient, using line search method
    [optP.tau0, J2, Gtau, exitflag, output] = costfunmin( u0, PR, optP.tau0, Gtau );

    % Update the number of times the function is called
    Feval = Feval + length(Gtau);
        
    %% Storing Values
    Tau(p+1, 1)  = optP.tau0;
    % For diagnostics, save results from line search
    Exit(p, 1)   = exitflag;
    Num(p, 1)    = Feval;
    bval(p, 1)   = bPR;
    CG(:, p)     = PR;
    gradS(:, p)  = del_J;
    % Diagnostics for minimization process
    if ( exitflag ~= 1 )
        fprintf('  PROBLEM: exitflag=%d \n', exitflag);
    end

    % Store values for the Polak Ribere method
    delk = del_J;
    pr   = PR;

    % Ensure new value is smaller than previous iteration. If not, force to exit 
    if J2 > J1
        J2 = J1;             % This will force it to exit optimization loop
        disp('   Must exit loop!!!')
        IC_1(:, p+1) = u0;   % Store initial condition
        J(p+1, 1)    = J2;   % Store functional value 
    else
        IC_1(:, p+1) = u0;                % Store initial condition
        u0           = u0 + optP.tau0*PR; % Update initial condition
        u0           = ProjConstr( u0 );
        J(p+1, 1)    = J2;                % Store functional value
    end

    %% Update initial condition    
    % Solve forward burgers problem
    [u, ~] = BurgersSolve( u0 );

    % Compute value of cost functional
    [J2, Jadj_hat, Jsimpl_hat] = costfunF( u0, u(:, end) );
    
    % Display the current iteration
    disp('========================================');
    fprintf('Iteration: %i, Cost Function: %d \n', p, J2)

    % Increment iteration counter
    p = p+1;    
end % End - While loop

disp('========================================');
disp(' Optimization terminated ');
disp('========================================');

%% Finalize and plot results
% Storing optimal values and diagnostic quantities
Exit  = Exit(1:p-1, 1);
Num   = Num(1:p-1, 1);
CG    = CG(1:length(uIC), 1:p-1);
bval  = bval(1:p-1, 1);
gradS = gradS(1:length(uIC), 1:p-1);
IC_1  = IC_1(1:length(uIC), 1:p);
J     = J(1:p, 1);
Tau   = Tau(1:p, 1);
%
u0_op = u0;

% Stop timing
time = toc;

% Display Optimization information
fprintf('\n Number of iterations: %i, Final J: %i, Time: %g seconds. \n', p, J(end), time)

% Plots results
pltFuncs( u0, u, J )
