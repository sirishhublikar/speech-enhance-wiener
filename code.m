%% Read the clean and noise signals
clc; clear; 
[s_clean, fs] = audioread('clean_speech.wav');
[n_stat, fs_ns] = audioread('speech_shaped_noise.wav');
[n_babble, fs_nb] = audioread('babble_noise.wav');


% Change sampling rate if different
if fs_ns ~= fs
    n_stat = resample(n_stat, fs, fs_ns);
end
if fs_nb ~= fs
    n_babble = resample(n_babble, fs, fs_nb);
end

% soundsc(s_clean, fs);       % play clean speech
% soundsc(n_stat, fs);        % play noise
% soundsc(n_babble, fs);      % play noise babble

%% Add noise - Babble or stationary speech-shaped
% noise = n_babble(1:length(s_clean));
noise = n_stat(1:length(s_clean));  % trim to same length

% scale noise to get desired SNR
power_s = sum(s_clean.^2);
power_n = sum(noise.^2);
desired_snr = 10;  % in dB
scale = sqrt(power_s / (power_n * 10^(desired_snr/10)));
n_scaled = scale * noise;

x_noisy = s_clean + n_scaled;

% soundsc(x_noisy, fs);  % listen

%% Sectrograms
% Parameters
win = hamming(1024);
noverlap = 512;
nfft = 2048;

[~, F, T, P_clean] = spectrogram(s_clean, win, noverlap, nfft, fs);
[~, ~, ~, P_noise] = spectrogram(n_scaled, win, noverlap, nfft, fs);
[~, ~, ~, P_noisy] = spectrogram(x_noisy, win, noverlap, nfft, fs);

t = (0:length(s_clean)-1)/fs;  % Time vector in seconds

%% Signal Estimation
% Estimate noise from silent segment in the noisy speech (first 0.25s)

% PSD of noisy speech
[S_noisy, F, T] = stft(x_noisy, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft);
Pxx = abs(S_noisy).^2;

% Select a silent segment for noise estimation
a = 0.25; % First few seconds of noisy speech
estimatedNoise = x_noisy(1:round(a * fs));

% PSD of the estimated noise
[S_noise, ~, ~] = stft(estimatedNoise, fs, 'Window', win,'OverlapLength', noverlap, 'FFTLength', nfft);
Pnn = mean(abs(S_noise).^2, 2);

Pnn_rep = repmat(Pnn, 1, size(Pxx,2)); 
Pss = max(Pxx - Pnn_rep, 0);              

% Wiener Filter
H = Pss ./ (Pss + 3.5*Pnn_rep);
% H = Pss ./ (Pss + Pnn);
S_hat = H .* S_noisy;

% Inverse STFT to reconstruct enhanced signal
x_hat = istft(S_hat, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft);

% SNR
L = min([length(s_clean), length(x_noisy), length(x_hat)]);
s_clean = s_clean(1:L);
x_noisy = x_noisy(1:L);
x_hat   = x_hat(1:L);

% Input SNR
noise_in = x_noisy - s_clean;
SNR_in = 10 * log10(sum(s_clean.^2) / sum(noise_in.^2));

% Output SNR
noise_out = x_hat - s_clean;
SNR_out = 10 * log10(sum(s_clean.^2) / sum(noise_out.^2));

% Improvement
SNR_improvement = SNR_out - SNR_in;

fprintf('Input SNR:  %.2f dB\n', SNR_in);
fprintf('Output SNR: %.2f dB\n', SNR_out);
fprintf('SNR Improvement: %.2f dB\n', SNR_improvement);

% soundsc(x_noisy, fs);       % play noisy speech
% soundsc(x_hat, fs);       % play estimated speech

%% Mean-Square Error (MSE)

L = min(length(s_clean), length(x_hat));
s_clean = s_clean(1:L);
x_hat   = x_hat(1:L);

% MSE
MSE = mean((s_clean - x_hat).^2);

% MSE in dB
MSE_dB = 10 * log10(MSE);

fprintf('Mean-Square Error (linear): %.6f\n', MSE);
fprintf('Mean-Square Error (dB): %.2f dB\n', MSE_dB);


%% Plot of results
figure;

subplot(3,1,1)
imagesc(T,F,10*log10(P_clean)); axis xy; colormap jet; colorbar;
title('Clean Speech'); ylim([0 fs/2]);

subplot(3,1,2);
imagesc(T,F,10*log10(P_noisy)); axis xy; colormap jet; colorbar;
title('Noisy Speech'); ylim([0 fs/2]);

subplot(3,1,3);
[~,~,~,P_hat] = spectrogram(x_hat, win, noverlap, nfft, fs);
imagesc(T,F,10*log10(P_hat)); axis xy; colormap jet; colorbar;
title('Enhanced Speech'); ylim([0 fs/2]);
xlabel('Time (s)'); ylabel('Frequency (Hz)');

sgtitle('Spectrograms')