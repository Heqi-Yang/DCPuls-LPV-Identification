clear; clc; close all;

snr = [20, 15, 10, 5];
n = 2;
r = 3;
N = 200;
Nmc = 100;
nSnr = numel(snr);
data = zeros(Nmc, nSnr);
data_vaf = zeros(1, nSnr);
data_nrmse = zeros(1, nSnr);
data_nrmse_box = zeros(Nmc, nSnr);
data_time = zeros(1, nSnr);
for no = 1:nSnr
    
    VAF_values = zeros(1, Nmc); 
    NRMSE_values = zeros(1, Nmc);
    time_values = zeros(1, Nmc);
    for mc = 1:Nmc
        
        fprintf('%d/%d\n',mc,Nmc);
        u = randn(1, N);  
        u_vali = randn(1,N);
        mu = make_mu(N);
        
        A_cell = cell(1, r);
        A_data = [4/15,  1/15,  3/20,  -1/60,  29/405,  2/81;
                 -1/6,  1/30,  -1/60,  3/20,  1/81,   52/405 ];
        
        A_cell{1} = A_data(:, 1:2);    % A^{(1)} 
        A_cell{2} = A_data(:, 3:4);    % A^{(2)}   
        A_cell{3} = A_data(:, 5:6);    % A^{(3)} 
        
        B_cell = cell(1, r);
        B_data = [1,  0.2,  0.2;
                  0,  0.2, -0.2];
        
        B_cell{1} = B_data(:, 1);      % B^{(1)}
        B_cell{2} = B_data(:, 2);      % B^{(2)} 
        B_cell{3} = B_data(:, 3);      % B^{(3)}

        C_cell = cell(1, r);
        C_data = [1, 0, 0, 0, 0, 0];
        C_cell{1} = C_data(:, 1:2);      
        C_cell{2} = C_data(:, 3:4);       
        C_cell{3} = C_data(:, 5:6);

        x0 = [0; 0];          
        [y, ~] = sim_lpv(A_cell, B_cell, C_cell, u, mu, x0);
        [y_vali, ~] = sim_lpv(A_cell, B_cell, C_cell, u_vali, mu, x0);
        % noise option
        useNoisyIdentification = true;
        signal_power = mean(y.^2);
        snr_db = snr(no);
        noise_power = signal_power / 10^(snr_db/10);
        noise = sqrt(noise_power) * randn(size(y));
            
        if useNoisyIdentification
            y_noisy = y + noise;
            noiseScale = 10^((20 - snr_db) / 6.05);
            noiseScale = min(max(noiseScale, 1e-3), 300);
            lowSnrWeight = min(max((noiseScale - 1) / 299, 0), 1);
        else
            y_noisy = y; %#ok<UNRCH>
            noiseScale = 1;
            lowSnrWeight = 0;
        end
        identificationTimer = tic;
        p = 3;
        lambda = 1;
        dcpulsOpts = struct('maxIterations', 20, 'tolerance', 1e-2);
        dcpulsOpts.phiGlobalRidgeFactor = 1e-2;
        if useNoisyIdentification
            noiseRidge = [0, 1e-2, 1e-1] * noiseScale;
            highSnrRidge = [1e-1, 2.9e-1, 2.9] / noiseScale;
            lowSnrBoost = lowSnrWeight * [1e-1, 3, 30];
            dcpulsOpts.phiLagRidge = noiseRidge + highSnrRidge + lowSnrBoost;
        end
        % identification solver
        [~, phi_hat, ~] = idf_yu( ...
            u, y_noisy, mu, p, n, lambda, dcpulsOpts);
        hankelOpts = struct('iterations', 4, 'blend', 0.8, 'shiftWeight', 0.5);
        [phiReal, ~] = denoise_phi(phi_hat, n, r, p, hankelOpts);
        hoKalmanOpts = struct('shiftWeight', 0.5, 'rightRidgeFactor', 1e-2);
        [A, B, C, ~] = realize_phi(phiReal, n, r, p, hoKalmanOpts);





















        time_values(mc) = toc(identificationTimer);

        %% Validation
        x_est = zeros(n, N-p);
        y_est = zeros(1, N-p);
        x_est(:,1) = x0;
        for i = 1:N-p
            x_est(:,i+1) = A * kron(mu(:,i), x_est(:,i)) + B * kron(mu(:,i), u_vali(i));
            y_est(i) = C * kron(mu(:,i), x_est(:,i));
        end
        VAF_est = calc_vaf(y_vali(1:N-p), y_est);
        NRMSE_est = calc_nrmse(y_vali(1:N-p), y_est);
        fprintf('VAF = %.2f\n', VAF_est);
        fprintf('NRMSE = %.2f\n', NRMSE_est);
        VAF_values(mc) = VAF_est;
        NRMSE_values(mc) = NRMSE_est;
    end
    VAF_mean = mean(VAF_values);
    NRMSE_mean = mean(NRMSE_values);
    fprintf('\nMean VAF = %.2f\n', VAF_mean);
    fprintf('Mean NRMSE = %.2f\n', NRMSE_mean)
    fprintf('Mean DCPuls time = %.3f s\n', mean(time_values));
    data(:, no) = VAF_values';
    data_vaf(no) = VAF_mean;
    data_nrmse(no) = NRMSE_mean;
    data_nrmse_box(:, no) = NRMSE_values';
    data_time(no) = mean(time_values);
end

%% Boxplot
figure('Color', 'w', 'Name', 'DCPuls performance distribution', ...
    'Position', [100, 100, 980, 420]);

subplot(1, 2, 1);
boxplot(data, ...
    'Labels', arrayfun(@(x) sprintf('%.3g', x), snr(:)', 'UniformOutput', false), ...
    'Symbol', 'k+', ...
    'Whisker', 1.5);
hold on;
plot(1:numel(snr), data_vaf, 'o-', ...
    'Color', [0.85, 0.20, 0.15], ...
    'MarkerFaceColor', [0.85, 0.20, 0.15], ...
    'MarkerEdgeColor', 'w', ...
    'LineWidth', 1.6, ...
    'MarkerSize', 6);
ylabel('VAF (%)');
xlabel('SNR (dB)');
title('DCPuls VAF under different noise levels');
grid on;
box on;
set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12, ...
    'LineWidth', 1, ...
    'GridAlpha', 0.18, ...
    'MinorGridAlpha', 0.10);
set(findobj(gca, 'Tag', 'Box'), 'LineWidth', 1.3, 'Color', [0.10, 0.30, 0.55]);
set(findobj(gca, 'Tag', 'Median'), 'LineWidth', 1.4, 'Color', [0.05, 0.05, 0.05]);
set(findobj(gca, 'Tag', 'Whisker'), 'LineWidth', 1.1, 'Color', [0.25, 0.25, 0.25]);
set(findobj(gca, 'Tag', 'Upper Whisker'), 'LineWidth', 1.1, 'Color', [0.25, 0.25, 0.25]);
set(findobj(gca, 'Tag', 'Lower Whisker'), 'LineWidth', 1.1, 'Color', [0.25, 0.25, 0.25]);
legend({'Mean VAF'}, 'Location', 'southwest', 'Box', 'off');

subplot(1, 2, 2);
boxplot(data_nrmse_box, ...
    'Labels', arrayfun(@(x) sprintf('%.3g', x), snr(:)', 'UniformOutput', false), ...
    'Symbol', 'k+', ...
    'Whisker', 1.5);
hold on;
plot(1:numel(snr), data_nrmse, 'o-', ...
    'Color', [0.85, 0.20, 0.15], ...
    'MarkerFaceColor', [0.85, 0.20, 0.15], ...
    'MarkerEdgeColor', 'w', ...
    'LineWidth', 1.6, ...
    'MarkerSize', 6);
ylabel('NRMSE (%)');
xlabel('SNR (dB)');
title('DCPuls NRMSE under different noise levels');
grid on;
box on;
set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 12, ...
    'LineWidth', 1, ...
    'GridAlpha', 0.18, ...
    'MinorGridAlpha', 0.10);
set(findobj(gca, 'Tag', 'Box'), 'LineWidth', 1.3, 'Color', [0.10, 0.30, 0.55]);
set(findobj(gca, 'Tag', 'Median'), 'LineWidth', 1.4, 'Color', [0.05, 0.05, 0.05]);
set(findobj(gca, 'Tag', 'Whisker'), 'LineWidth', 1.1, 'Color', [0.25, 0.25, 0.25]);
set(findobj(gca, 'Tag', 'Upper Whisker'), 'LineWidth', 1.1, 'Color', [0.25, 0.25, 0.25]);
set(findobj(gca, 'Tag', 'Lower Whisker'), 'LineWidth', 1.1, 'Color', [0.25, 0.25, 0.25]);
legend({'Mean NRMSE'}, 'Location', 'northeast', 'Box', 'off');

dcpuls_results = struct();
dcpuls_results.method = 'DCPuls';
dcpuls_results.snr = snr(:)';
dcpuls_results.meanVaf = data_vaf(:)';
dcpuls_results.meanNrmse = data_nrmse(:)';
dcpuls_results.meanTime = data_time(:)';

%% =================SAVE======================
save('DCPuls_snr_results.mat', 'dcpuls_results');

function mu = make_mu(N)
    randomChannels = 0.8 * (2 * rand(2, N) - 1);
    mu2 = randomChannels(1, :);
    mu3 = randomChannels(2, :);
    mu = [ones(1, N); mu2; mu3];
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
cols = rows;
oneset = words(r, 1);
ratio = nan(1, nIter);

for iter = 1:nIter
    [H, Hs] = hankel_blocks(phiOut, rows, cols, oneset, r);
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
    [sums, counts] = add_hankel(Hr(:, 1:size(H, 2)), rows, cols, ...
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
