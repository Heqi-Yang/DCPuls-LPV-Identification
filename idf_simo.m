function X_hat = idf_simo(u, y, mu, p, n, lambda)
    
    [~, N] = size(u);
    [l, ~] = size(y);
    [r, ~] = size(mu);
    
    [O_test, ~, ~] = build_matrices(u, y, mu, 1, p, l, r);
    nx = size(O_test, 2); 
    
    %% 1. 预计算阶段
    Data_O = cell(1, N-p);
    Data_U = cell(1, N-p);
    Data_y = cell(1, N-p);
    
    for k = 1:N-p
        [O_pt, U_pt, y_bar] = build_matrices(u, y, mu, k, p, l, r);
        Data_O{k} = O_pt;
        Data_U{k} = U_pt;
        Data_y{k} = y_bar;
    end
    
    %% 2. 初始估计
    Weight_Matrix = eye(nx); 

    [X_bar, ~] = solve_step_noise_resistant(Data_O, Data_U, Data_y, nx, lambda, Weight_Matrix, N, p);
    
    %% 3. 迭代优化
    max_iter = 20; 
    tol = 1;    
    
    for iter = 1:max_iter
        [U_svd, ~, ~] = svd(X_bar, 'econ');
        Q_k = U_svd(:, 1:n);
        Q_perp = eye(size(Q_k,1)) - Q_k * Q_k';
        
        X_bar_old = X_bar;
        
        [X_bar_new_raw, ~] = solve_step_noise_resistant(Data_O, Data_U, Data_y, nx, lambda, Q_perp, N, p);
        
        X_bar_new = X_bar_new_raw ;

        diff_norm = norm(X_bar_old - X_bar_new, 'fro') / (norm(X_bar_old, 'fro') + 1e-6);
        
        if diff_norm < tol
            X_bar = X_bar_new;
            break;
        else
            X_bar = X_bar_new;
        end
    end
    
    [~, ~, V_final] = svd(X_bar, 'econ');
    X_hat = V_final(:, 1:n)';
end

%% ===
function [X_bar, phi] = solve_step_noise_resistant(Data_O, Data_U, Data_y, nx, lambda, Weight_Block, N, p)
    
    R11_cell = cell(1, N-p);
    R12_cell = cell(1, N-p);
    R13_cell = cell(1, N-p);
    R22_cell = cell(1, N-p);
    R23_cell = cell(1, N-p);
    
    sqrt_lambda = sqrt(lambda);
    
    for k = 1:N-p
        O_pt = Data_O{k};
        U_pt = Data_U{k};
        y_bar = Data_y{k};
        
        current_nx = size(O_pt, 2); 
        
        Reg_Row = [sqrt_lambda * Weight_Block, zeros(current_nx, size([O_pt, U_pt, y_bar],2)-current_nx)];
        Aug_t = [O_pt, U_pt, y_bar; Reg_Row];
        
        [~, R_t] = qr(Aug_t, 0);
        
        R11 = R_t(1:current_nx, 1:current_nx); 
        R12 = R_t(1:current_nx, current_nx+1:end-1);
        R13 = R_t(1:current_nx, end);
        
        R22 = R_t(current_nx+1:end, current_nx+1:end-1);
        R23 = R_t(current_nx+1:end, end);
        
        R11_cell{k} = R11;
        R12_cell{k} = R12;
        R13_cell{k} = R13;
        R22_cell{k} = R22;
        R23_cell{k} = R23;
    end
    
    R22_stack = vertcat(R22_cell{:}); 
    R23_stack = cell2mat(R23_cell'); 
    
    % Global QR solve, consistent with idf_yu.  This avoids forming the
    % squared normal matrix R22' * R22 used by the former ridge solve.
    nPhi = size(R22_stack, 2);
    [~, R2223] = qr([R22_stack, R23_stack], 0);
    Rf1 = R2223(1:nPhi, 1:nPhi);
    Rf2 = R2223(1:nPhi, end);
    phi = Rf1 \ Rf2;

    X_bar = zeros(nx, N-p);
    for t = 1:N-p
        if isempty(phi)
             X_bar(:, t) = R11_cell{t} \ R13_cell{t};
        else
             X_bar(:, t) = R11_cell{t} \ (R13_cell{t} - R12_cell{t} * phi);
        end
    end
end

%% ===
function [O_pt, U_pt, y_bar] = build_matrices(u, y, mu, k, p, l, r)
    y_bar = zeros(size(y,1)*(p+1), 1); 
    
    Gamma_O = cell(1, p+1);
    for i = 1:p+1
        y_bar((i-1)*l+1 : i*l, :) = y(:, k+i-1);
        
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
            Gamma_U{row} = zeros(l, dim_gamma_j * l * size(u,1)); 
        end
        for row = j : (p + 1)
            i = row - j; 
            gamma = 1;
            for k_step = (j-1) : -1 : 0
                gamma = kron(gamma, mu(:, k+i+k_step));
            end
            
            gu = kron(u(:, k+i)', gamma); 
            
            Gamma_U{row} = kron(gu', eye(l));
        end
        U_cols{j-1} = vertcat(Gamma_U{:});
    end
    U_pt = horzcat(U_cols{:});
end
