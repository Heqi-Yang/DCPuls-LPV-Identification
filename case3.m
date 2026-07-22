clear; clc; close all;
%% Parameter
n = 3;  % State
m = 1;  % Input
l = 2;  % Output
r = 3;  % Scheduling variable
N = 500;  % Datapoint
Nmc = 100;%Monte Carlo
VAF_values = zeros(1, Nmc); 
NRMSE_values = zeros(1, Nmc);
for mc = 1:Nmc
    fprintf('%d/%d\n',mc,Nmc);
    u = randn(1, N); 
    u_vali =  randn(1,N);

    mu = make_mu(r, N);
    A_cell = cell(1, r);
    A1 = [  4/10,   1/10,   0.00;
                  1/6,    1/30,   0.10;   
                   0.00,   0.05,   3/5 ];  
    
    A2 = [  3/20,  -1/6,   0.00;
                  1/6,   3/20,   1/4;   
                   0.00,  -1/10,   1/5 ];
    A3 = [  1/15,   1/4,   0.00;
                   1/4,   1/8,    0.00;
                   0.00,   0.00,   1/10 ]; 

    A_cell{1} = A1;
    A_cell{2} = A2;
    A_cell{3} = A3;
    B_cell = cell(1, r);

    B_cell{1} = [ 1.0; 
                  0.0; 
                  0.0 ];
    
    B_cell{2} = [ 0.2; 
                  0; 
                  0 ];
    
    B_cell{3} = [ 0.2; 
                 -0; 
                  0 ];
    C_cell = cell(1, r);
    C = [ 1, 0, 0,0,0,0,0,0,0;
            0,0,1,0,0,1,0,0,1];   
    C_cell{1} = C(:, 1:3);
    C_cell{2} = C(:, 4:6);
    C_cell{3} = C(:, 7:9);
    %% System
    x0 = [0; 0; 0];  
    [y, ~] = sim_lpv(A_cell, B_cell, C_cell, u, mu, x0);
    [y_vali, ~] = sim_lpv(A_cell, B_cell, C_cell, u_vali, mu, x0);
    
    
    snr_db = 10; 
    snr_linear = 10^(snr_db/10);
    noise = zeros(size(y));
    
    for i = 1:l
        signal_power_i = mean(y(i, :).^2);
        noise_power_i = signal_power_i / snr_linear;
        noise(i, :) = sqrt(noise_power_i) * randn(1, N);
    end

    y_noisy = y + noise;
    
    
    tic;
    p = 3;
    lambda = 0.1;
    X_hat = idf_simo(u, y_noisy, mu, p, n, lambda);

    len = size(X_hat, 2) - 1;
    Xnext = X_hat(:, 2:end);
    regAB = zeros(n*r + m*r, len);
    for k = 1:len
        regAB(:, k) = [kron(mu(:, k), X_hat(:, k)); kron(mu(:, k), u(:, k))];
    end
    AB = Xnext * pinv(regAB);
    A = AB(:, 1:n*r);
    B = AB(:, n*r+1:end);

    regC = zeros(n*r, len);
    for k = 1:len
        regC(:, k) = kron(mu(:, k), X_hat(:, k));
    end
    C = y_noisy(:, 1:len) * pinv(regC);
    toc;
    
    %% Validation
    x_est = zeros(n, N-p+1);
    y_est = zeros(l, N-p);
    x_est(:,1) = x0;
    for i = 1:N-p
        x_est(:,i+1) = A * kron(mu(:,i), x_est(:,i)) + B * kron(mu(:,i), u_vali(i));
        y_est(:, i) = C * kron(mu(:,i), x_est(:,i));
    end
    x_est = x_est(:,1:end-1);

    
    %% VAF
    VAF_est = calc_vaf(y_vali(1, 1:N-p), y_est(1,:));
    NRMSE_est = calc_nrmse(y_vali(1, 1:N-p), y_est(1,:));
    fprintf('VAF = %.1f\n', VAF_est);
    fprintf('Output NRMSE = %.1f%%\n', NRMSE_est);
    VAF_values(mc) = VAF_est;
    NRMSE_values(mc) = NRMSE_est;
end
VAF_mean = mean(VAF_values);
NRMSE_mean = mean(NRMSE_values);
fprintf('Mean VAF = %.1f\n', VAF_mean);
fprintf('Mean output NRMSE = %.1f%%\n', NRMSE_mean);
fprintf('Std output NRMSE = %.1f%%\n', std(NRMSE_values));
%% Boxplot
figure;
boxplot(VAF_values);
ylabel('VAF (%)');
xlabel('Monte Carlo Trials');
title('Monte Carlo Simulation of VAF');
grid on;

figure;
boxplot(NRMSE_values);
ylabel('Output NRMSE (%)');
xlabel('Monte Carlo Trials');
title('Monte Carlo Simulation of Output NRMSE');
grid on;
%% function
% 
function mu = make_mu(r, N)
    mu = zeros(r, N);
    mu(1, :) = 1;
  
    % Independent, bounded scheduling channels provide persistent
    % excitation for the high-order Kronecker regressors in idf_simo.
    randomChannels = 2 * rand(2, N) - 1;
    mu(2, :) = 0.2 + 0.2 * randomChannels(1, :);
    mu(3, :) = 0.4 * randomChannels(2, :);

end

function [y, x] = sim_lpv(A_cell, B_cell, C_cell, u, mu, x0)
    
    N = length(u);
    n = size(A_cell{1}, 1);
    l = size(C_cell{1}, 1);
    x = zeros(n, N+1);
    y = zeros(l, N);
    
    x(:, 1) = x0;
    
    for k = 1:N
        A_mu = zeros(n, n);
        B_mu = zeros(n, 1);
        C_mu = zeros(l, n);
        for i = 1:length(A_cell)
            A_mu = A_mu + mu(i, k) * A_cell{i};
            B_mu = B_mu + mu(i, k) * B_cell{i};
            C_mu = C_mu + mu(i, k) * C_cell{i};
        end
        
        x(:, k+1) = A_mu * x(:, k) + B_mu * u(k);
        y(:,k) = C_mu * x(:, k);
    end
    
    x = x(:, 1:end-1); 
end

function value = calc_vaf(y, yhat)
value = max(0, 1 - var(y - yhat) / var(y)) * 100;
end

function value = calc_nrmse(y, yhat)
if ~isequal(size(y), size(yhat))
    error('y and yhat must have the same dimensions.');
end
y = y(:);
yhat = yhat(:);
value = 100 * norm(y - yhat) / norm(y - mean(y));
end
