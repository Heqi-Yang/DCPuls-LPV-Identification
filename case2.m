clear; clc; close all;
%% Parameter
n = 4;  % State
r = 2;  % Scheduling variable
N = 200;  % Datapoint
Pi = 20;  % Scheduling parameter
Nmc = 100;%Monte Carlo
VAF_values = zeros(1, Nmc); 
NRMSE_values = zeros(1, Nmc);
time_values = zeros(1, Nmc);
for mc = 1:Nmc
    fprintf('%d/%d\n',mc,Nmc);
    u = 0.5*randn(1, N);  % White noise
    u_vali = 0.5* randn(1,N);
    mu = make_mu(r, N, Pi);
    
    A_cell = cell(1, r);
    A_data = [-1/30,  1/30,  11/75,  0,  -3/10,  1/6, 11/30, 0;
         0,  1/20,  -3/20, 3/20, 0, 7/60, -1/20, 1/20;
         0, 1/10, 32/75, 0, 0, 3/20, 31/60,0;
         0, 0, 0, 23/30, 0, 0, 0, 1/10];
    A_cell{1} = A_data(:, 1:4);    % A^{(1)} 
    A_cell{2} = A_data(:, 5:8);    % A^{(2)}   

    
    B_cell = cell(1, r);
    B_data = [1, 0.2;
              0, 0.2;
              0, 0.2;
              0, 0.2];
    
    B_cell{1} = B_data(:, 1);      % B^{(1)}
    B_cell{2} = B_data(:, 2);      % B^{(2)} 

    C_cell = cell(1,r);
    C_cell{1} = [1, 0, 0, 0];
    C_cell{2} = [0, 0, 0, 0];
    %% System
    x0 = [0; 0; 0; 0];  
    [y, ~] = sim_lpv(A_cell, B_cell, C_cell, u, mu, x0);
    [y_vali, ~] = sim_lpv(A_cell, B_cell, C_cell, u_vali, mu, x0);
    
    signal_power = mean(y.^2);
        
    snr_db = 10;
    snr_linear = 10^(snr_db/10);  
    noise_power = signal_power / snr_linear;
    noise = sqrt(noise_power) * randn(size(y));
        
    y_noisy = y + noise;

    
    %% DCPuls identification: phi estimation, Hankel denoising, Ho-Kalman realization
    identificationTimer = tic;
    p = 6;
    lambda = 1;
    noiseScale = min(max(10^((20 - snr_db) / 6.05), 1e-3), 1000);
    dcpulsOpts = struct('maxIterations', 20, 'tolerance', 1e-4, ...
        'useParallel', true);
    dcpulsOpts.phiLagRidge = [1e-1, 3e-1, 1, 3, 10, 30] * noiseScale;
    [~, phi_hat, dcpulsInfo] = idf_yu( ...
        u, y_noisy, mu, p, n, lambda, dcpulsOpts);
    hankelOpts = struct('iterations', 4, 'blend', 0.8, 'shiftWeight', 0.5);
    [phiReal, hankelInfo] = denoise_phi(phi_hat, n, r, p, hankelOpts);
    hoKalmanOpts = struct('shiftWeight', 0.5, 'rightRidgeFactor', 1e-2);
    [A, B, C, hoInfo] = realize_phi(phiReal, n, r, p, hoKalmanOpts);
    time_values(mc) = toc(identificationTimer);
    
    %% Validation
    x_est = zeros(n, N-p+1);
    x_est(:,1) = x0;
    y_est = zeros(1, N-p);
    for i = 1:N-p
        x_est(:,i+1) = A * kron(mu(:,i), x_est(:,i)) + B * kron(mu(:,i), u_vali(i));
        y_est(i) = C * kron(mu(:,i), x_est(:,i));
    end
    x_est = x_est(:,1:end-1);

    VAF_est = calc_vaf(y_vali(1:N-p), y_est);
    NRMSE_est = calc_nrmse(y_vali(1:N-p), y_est);
    fprintf('VAF = %.1f\n', VAF_est);
    fprintf('Output NRMSE = %.1f%%\n', NRMSE_est);
    VAF_values(mc) = VAF_est;
    NRMSE_values(mc) = NRMSE_est;
end
VAF_mean = mean(VAF_values);
NRMSE_mean = mean(NRMSE_values);
fprintf('Mean VAF = %.1f\n', VAF_mean);
fprintf('Mean output NRMSE = %.1f%%\n', NRMSE_mean);
fprintf('Std  output NRMSE = %.1f%%\n', std(NRMSE_values));
fprintf('Mean DCPuls time = %.3f s\n', mean(time_values));
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
function mu = make_mu(r, N, Pi)

    % μ_k^(2) = cos(2πk*Π/N)/2 + 0.2
    % μ_k^(1) = 1
    
    k = 1:N;
    mu = zeros(r, N);
    
    % μ_k^(1) = 1
    mu(1, :) = 1;
    
    % μ_k^(2)
    mu(2, :) = cos(2*pi*k*Pi/N)/5 + 0.2;


end

function [y, x] = sim_lpv(A_cell, B_cell, C_cell, u, mu, x0)
    
    N = length(u);
    n = size(A_cell{1}, 1);
    x = zeros(n, N+1);
    y = zeros(1, N);
    
    x(:, 1) = x0;
    
    for k = 1:N
        A_mu = zeros(n, n);
        B_mu = zeros(n, 1);
        C_mu = zeros(1, n);
        for i = 1:length(A_cell)
            A_mu = A_mu + mu(i, k) * A_cell{i};
            B_mu = B_mu + mu(i, k) * B_cell{i};
            C_mu = C_mu + mu(i, k) * C_cell{i};
        end
        x(:, k+1) = A_mu * x(:, k) + B_mu * u(k);
        y(k) = C_mu * x(:, k);
    end
    
    x = x(:, 1:end-1); 
end

function [phiOut, info] = denoise_phi(phi, n, r, p, opts)
if p < 3
    error('At least p = 3 is required for the generalized Hankel projection.');
end
nIter = get_opt(opts, 'iterations', 4);
blend = min(max(get_opt(opts, 'blend', 0.8), 0), 1);
weight = max(get_opt(opts, 'shiftWeight', 0.5), eps);
phiOut = phi(:);
rows = [words(r, 1), zeros(r, 1); words(r, 2)];
oneset = words(r, 1);
ratio = nan(1, nIter);
for iter = 1:nIter
    [H, Hs] = hankel_blocks(phiOut, rows, rows, oneset, r);
    Hj = H;
    for mode = 1:r
        Hj = [Hj, sqrt(weight) * Hs{mode}]; %#ok<AGROW>
    end
    [U, S, V] = svd(Hj, 'econ');
    sv = diag(S);
    if numel(sv) > n
        ratio(iter) = sv(n + 1) / max(sv(n), eps);
    end
    Hr = U(:, 1:n) * S(1:n, 1:n) * V(:, 1:n)';
    sums = zeros(size(phiOut));
    counts = zeros(size(phiOut));
    [sums, counts] = add_hankel(Hr(:, 1:size(H, 2)), rows, rows, ...
        sums, counts, r, []);
    for mode = 1:r
        first = size(H, 2) + (mode - 1) * size(oneset, 1) + 1;
        block = Hr(:, first:first + size(oneset, 1) - 1) / sqrt(weight);
        [sums, counts] = add_hankel(block, rows, oneset, ...
            sums, counts, r, mode);
    end
    projected = sums ./ max(counts, 1);
    phiOut = (1 - blend) * phiOut + blend * projected;
end
info = struct('iterations', nIter, 'blend', blend, 'shiftWeight', weight, ...
    'rankRatioHistory', ratio, ...
    'relativeCorrection', norm(phiOut - phi(:)) / max(norm(phi(:)), eps));
end

function [A, B, C, info] = realize_phi(phi, n, r, p, opts)
if p < 3
    error('At least p = 3 is required for this Ho-Kalman realization.');
end
rows = [words(r, 1), zeros(r, 1); words(r, 2)];
oneset = words(r, 1);
[H, Hs] = hankel_blocks(phi(:), rows, rows, oneset, r);
weight = max(get_opt(opts, 'shiftWeight', 0.5), eps);
Hj = H;
for mode = 1:r
    Hj = [Hj, sqrt(weight) * Hs{mode}]; %#ok<AGROW>
end
[U, S, V] = svd(Hj, 'econ');
sv = diag(S);
if numel(sv) < n || sv(n) <= eps(max(sv))
    error('Generalized Hankel matrix does not have numerical rank n.');
end
if numel(sv) > n
    rankRatio = sv(n + 1) / max(sv(n), eps);
else
    rankRatio = 0;
end
Obs = U(:, 1:n) * sqrt(S(1:n, 1:n));
Ctr = sqrt(S(1:n, 1:n)) * V(:, 1:n)';
R1 = Ctr(:, 1:r);
gram = R1 * R1';
noiseLevel = min(1, rankRatio / max(get_opt(opts, 'robustRankRatio', 2e-2), eps));
ridge = get_opt(opts, 'rightRidgeFactor', 0) * noiseLevel * trace(gram) / n;
C = reshape(Obs(1:r, :)', 1, []);
B = R1;
A = zeros(n, n * r);
shiftErr = zeros(r, 1);
for mode = 1:r
    first = size(H, 2) + (mode - 1) * r + 1;
    Rs = Ctr(:, first:first + r - 1) / sqrt(weight);
    if ridge > 0
        Am = Rs * R1' / (gram + ridge * eye(n));
    else
        Am = Rs / R1;
    end
    A(:, (mode - 1) * n + (1:n)) = Am;
    shiftErr(mode) = norm(Hs{mode} - Obs * Am * R1, 'fro') / ...
        max(norm(Hs{mode}, 'fro'), eps);
end
info = struct('singularValues', sv, 'rankRatio', rankRatio, ...
    'shiftResidual', shiftErr, 'hankel', H, 'jointHankel', Hj, ...
    'shiftWeight', weight, 'rightRidge', ridge, ...
    'noiseLevel', noiseLevel, 'rightCondition', cond(R1));
end

function [H, Hs] = hankel_blocks(phi, rows, cols, oneset, r)
H = zeros(size(rows, 1), size(cols, 1));
for i = 1:size(rows, 1)
    a = trim_word(rows(i, :));
    for j = 1:size(cols, 1)
        H(i, j) = phi(phi_index([a, trim_word(cols(j, :))], r));
    end
end
Hs = cell(1, r);
for mode = 1:r
    Hs{mode} = zeros(size(rows, 1), size(oneset, 1));
    for i = 1:size(rows, 1)
        a = trim_word(rows(i, :));
        for j = 1:size(oneset, 1)
            Hs{mode}(i, j) = phi(phi_index([a, mode, oneset(j, :)], r));
        end
    end
end
end

function [sums, counts] = add_hankel(block, rows, cols, sums, counts, r, mode)
for i = 1:size(block, 1)
    a = trim_word(rows(i, :));
    for j = 1:size(block, 2)
        b = trim_word(cols(j, :));
        if isempty(mode)
            word = [a, b];
        else
            word = [a, mode, b];
        end
        idx = phi_index(word, r);
        sums(idx) = sums(idx) + block(i, j);
        counts(idx) = counts(idx) + 1;
    end
end
end

function table = words(r, len)
table = zeros(r^len, len);
for row = 1:size(table, 1)
    value = row - 1;
    for col = len:-1:1
        table(row, col) = mod(value, r) + 1;
        value = floor(value / r);
    end
end
end

function word = trim_word(word)
word = word(word > 0);
end

function idx = phi_index(word, r)
len = numel(word);
if len < 2
    error('Predictor phi contains Markov words of length two or greater.');
end
idx = sum(r .^ (2:len - 1)) + 1;
for col = 1:len
    idx = idx + (word(col) - 1) * r^(len - col);
end
end

function value = get_opt(opts, name, defaultValue)
if isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = defaultValue;
end
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

