function [X_hat, phi, info] = idf_yu(u, y, mu, p, n, lambda, opts)
%IDF_YU DCPuls implementation of Algorithm 1 in Yu, Yang, Verhaegen.

    if nargin < 7 || isempty(opts)
        opts = struct();
    end
    % Window-wise QR factorizations are independent. Use MATLAB's thread
    % pool by default; callers can still set opts.useParallel = false.
    useParallel = get_opt(opts, 'useParallel', true);
    if useParallel
        try
            if isempty(gcp('nocreate'))
                parpool('threads');
            end
        catch parallelError
            warning('idf_yu:ParallelUnavailable', ...
                'Parallel QR is unavailable (%s). Falling back to serial execution.', ...
                parallelError.message);
            useParallel = false;
        end
    end
    [m, N] = size(u);
    [l, ~] = size(y);
    [r, ~] = size(mu);
    nx = m*r*(1-r^(p+1))/(1-r);

    nw = N - p;
    Odata = cell(1, nw);
    Udata = cell(1, nw);
    ydata = cell(1, nw);
    for k = 1:nw
        [Odata{k}, Udata{k}, ydata{k}] = build_data(u, y, mu, k, p, l, r);
    end
    nPhi = size(Udata{1}, 2);
    zeroReg = zeros(nx, nPhi + 1);
    sqrtLambda = sqrt(lambda);
    [R11, R12, R13, R22, R23] = qr_windows( ...
        Odata, Udata, ydata, sqrtLambda * eye(nx), zeroReg, nx, useParallel);
    R22stack = vertcat(R22{:}); 
    R23stack = cell2mat(R23'); 
    phiLagRidge = get_opt(opts, 'phiLagRidge', zeros(1, p));
    phiGlobalRidgeFactor = get_opt(opts, 'phiGlobalRidgeFactor', 0);
    phi = solve_phi(R22stack, R23stack, r, p, ...
        phiLagRidge, phiGlobalRidgeFactor);
    Xbar = zeros(nx, nw);
    for t = 1:nw
        Xbar(:, t) = R11{t} \ (R13{t} - R12{t} * phi);        
    end

    [U0, S0, V0] = svd(Xbar, 'econ');
    Xbar = U0(:, 1:n) * S0(1:n, 1:n) * V0(:, 1:n)';

    maxIter = get_opt(opts, 'maxIterations', 100);
    tol = get_opt(opts, 'tolerance', 1e-7);
    
    relativeChange = inf;
    for iter = 1:maxIter
        [Ux, ~, ~] = svd(Xbar, 'econ');
        Q = Ux(:, 1:n);
        Qperp = eye(nx) - Q * Q';
      
        [R11, R12, R13, R22, R23] = qr_windows( ...
            Odata, Udata, ydata, sqrtLambda * Qperp, zeroReg, nx, useParallel);
        R22stack = vertcat(R22{:}); 
        R23stack = cell2mat(R23'); 
        phi = solve_phi(R22stack, R23stack, r, p, ...
            phiLagRidge, phiGlobalRidgeFactor);
        Xnew = zeros(nx, nw);
        for t = 1:nw
            Xnew(:, t) = R11{t} \ (R13{t} - R12{t} * phi);        
        end

        relativeChange = norm(Xbar - Xnew, 'fro') / (norm(Xbar, 'fro') + eps);
        Xbar = Xnew;
        if relativeChange < tol
            break
        end
    end

    [~, S, V] = svd(Xbar, 'econ');
    X_hat = V(:, 1:n)';
    info = struct('iterations', iter, 'relativeChange', relativeChange, ...
        'stateSingularValues', diag(S), ...
        'liftedState', Xbar, 'usedParallel', useParallel);
end

function [R11, R12, R13, R22, R23] = qr_windows( ...
        Odata, Udata, ydata, reg, zeroReg, nx, useParallel)
    nw = numel(Odata);
    R11 = cell(1, nw);
    R12 = cell(1, nw);
    R13 = cell(1, nw);
    R22 = cell(1, nw);
    R23 = cell(1, nw);

    if useParallel
        parfor t = 1:nw
            [R11{t}, R12{t}, R13{t}, R22{t}, R23{t}] = ...
                qr_window(Odata{t}, Udata{t}, ydata{t}, reg, zeroReg, nx);
        end
    else
        for t = 1:nw
            [R11{t}, R12{t}, R13{t}, R22{t}, R23{t}] = ...
                qr_window(Odata{t}, Udata{t}, ydata{t}, reg, zeroReg, nx);
        end
    end
end

function [R11, R12, R13, R22, R23] = qr_window(O, U, y, reg, zeroReg, nx)
    R = qr([O, U, y; reg, zeroReg], 0);
    R11 = R(1:nx, 1:nx);
    R12 = R(1:nx, nx+1:end-1);
    R13 = R(1:nx, end);
    R22 = R(nx+1:end, nx+1:end-1);
    R23 = R(nx+1:end, end);
end

function value = get_opt(opts, name, defaultValue)
    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = defaultValue;
    end
end

function phi = solve_phi( ...
        R22_stack, R23_stack, r, p, lagRidge, globalRidgeFactor)
    n_phi = size(R22_stack, 2);
    if nargin < 5 || isempty(lagRidge)
        lagRidge = zeros(1, p);
    end
    if nargin < 6 || isempty(globalRidgeFactor)
        globalRidgeFactor = 0;
    end
    if numel(lagRidge) ~= p
        error('phiLagRidge must have p entries for Gamma_2 through Gamma_(p+1).');
    end

    if all(lagRidge == 0) && globalRidgeFactor == 0
        [~, R2223] = qr([R22_stack, R23_stack], 0);
        Rf1 = R2223(1:n_phi, 1:n_phi);
        Rf2 = R2223(1:n_phi, end);
        phi = Rf1 \ Rf2;
        return
    end

    penalty = zeros(n_phi, 1);
    first = 1;
    for lagIndex = 1:p
        last = first + r^(lagIndex + 1) - 1;
        penalty(first:last) = lagRidge(lagIndex);
        first = last + 1;
    end
    scale = sum(R22_stack(:).^2) / max(n_phi, 1);
    diagonalRidge = max(scale, eps) * (penalty + globalRidgeFactor);
    % Solve the regularized least-squares problem by QR, rather than by
    % forming R22' * R22, to preserve the numerical QR route of idf_yu.
    augmentedRegressor = [R22_stack; diag(sqrt(diagonalRidge))];
    augmentedTarget = [R23_stack; zeros(n_phi, size(R23_stack, 2))];
    [~, Raugmented] = qr([augmentedRegressor, augmentedTarget], 0);
    phi = Raugmented(1:n_phi, 1:n_phi) \ Raugmented(1:n_phi, end);
end

function [O_pt, U_pt, y_bar] = build_data(u, y, mu, k, p, l, r)
    y_bar = zeros(size(y,1)*(p+1), 1); 
    
    Gamma_O = cell(1, p+1);
    for i = 1:p+1
        y_bar((i-1)*l+1 : i*l, :) = y(k+i-1);
        gamma_T = 1; 
        for k_idx = (i-1) : -1 : 0
            gamma_T = kron(gamma_T, mu(:, k + k_idx)');
        end
        Gamma_O{i} = kron(gamma_T, eye(l));
    end
    O_pt = blkdiag(Gamma_O{:});
    
    U_cols = cell(1, p);
    for j = 2 : (p + 1)
        dim_gamma_j = r^j; 
        Gamma_U = cell(p + 1, 1);
        for row = 1 : (j - 1)
            Gamma_U{row} = zeros(l, dim_gamma_j * l);
        end
        for row = j : (p + 1)
            i = row - j; 
            gamma = 1;
            for k_step = (j-1) : -1 : 0
                gamma = kron(gamma, mu(:, k+i+k_step));
            end
            gu = gamma * u(k + i); 
            Gamma_U{row} = kron(gu', eye(l));
        end
        U_cols{j-1} = vertcat(Gamma_U{:});
    end
    U_pt = horzcat(U_cols{:});
end
