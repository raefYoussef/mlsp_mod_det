close all

clear;
clc;

%%% FSK %%%
M = 2; % Modulation order
freqsep = 8; % Frequency separation (Hz)
nsamp = 8; % Number of samples per symbol
Fs = 32; % Sample rate (Hz)
x = randi([0 M-1], 1000, 1); % Digital signal

y_fsk = fskmod(x, M, freqsep, nsamp, Fs);
y_fsk = y_fsk';
y_fsk = awgn(y_fsk, 20);
y_fsk_fft = fftshift(fft(y_fsk));
fsk_freqs = (-Fs / 2:Fs/length(y_fsk_fft):Fs / 2);

%%% BPSK %%%
y_bpsk = pskmod(x, M);
y_bpsk = y_bpsk';
y_bpsk = awgn(y_bpsk, 20);
y_bpsk2 = y_bpsk.^2;
y_bpsk4 = y_bpsk.^4;
y_bpsk8 = y_bpsk.^8;
y_bpsk_fft = fftshift(fft(y_bpsk));
y_bpsk_fft2 = fftshift(fft(y_bpsk2));
y_bpsk_fft4 = fftshift(fft(y_bpsk4));
y_bpsk_fft8 = fftshift(fft(y_bpsk8));
bpsk_freqs = (-Fs / 2:Fs/length(y_bpsk_fft):Fs / 2);

%%% QPSK %%%
M = 4; % Modulation order
x = randi([0 M-1], 1000, 1); % Digital signal

y_qpsk = pskmod(x, M);
y_qpsk = y_qpsk';
y_qpsk = awgn(y_qpsk, 20);
y_qpsk2 = y_qpsk.^2;
y_qpsk4 = y_qpsk.^4;
y_qpsk8 = y_qpsk.^8;
y_qpsk_fft = fftshift(fft(y_qpsk));
y_qpsk_fft2 = fftshift(fft(y_qpsk2));
y_qpsk_fft4 = fftshift(fft(y_qpsk4));
y_qpsk_fft8 = fftshift(fft(y_qpsk8));
qpsk_freqs = (-Fs / 2:Fs/length(y_qpsk_fft):Fs / 2);

%%% 8PSK %%%
M = 8; % Modulation order
x = randi([0 M-1], 1000, 1); % Digital signal

y_psk8 = pskmod(x, M);
y_psk8 = y_psk8';
y_psk8 = awgn(y_psk8, 20);
y_psk82 = y_psk8.^2;
y_psk84 = y_psk8.^4;
y_psk88 = y_psk8.^8;
y_psk8_fft = fftshift(fft(y_psk8));
y_psk8_fft2 = fftshift(fft(y_psk82));
y_psk8_fft4 = fftshift(fft(y_psk84));
y_psk8_fft8 = fftshift(fft(y_psk88));
psk8_freqs = (-Fs / 2:Fs/length(y_psk8_fft):Fs / 2);

%%% 8QAM %%%
y_qam8 = qammod(x, M);
y_qam8 = y_qam8';
y_qam8 = awgn(y_qam8, 20);
y_qam82 = y_qam8.^2;
y_qam84 = y_qam8.^4;
y_qam88 = y_qam8.^8;
y_qam8_fft = fftshift(fft(y_qam8));
y_qam8_fft2 = fftshift(fft(y_qam82));
y_qam8_fft4 = fftshift(fft(y_qam84));
y_qam8_fft8 = fftshift(fft(y_qam88));
qam8_freqs = (-Fs / 2:Fs/length(y_qam8_fft):Fs / 2);

%%% 16QAM %%%
M = 16; % Modulation order
x = randi([0 M-1], 1000, 1); % Digital signal

y_qam16 = qammod(x, M);
y_qam16 = y_qam16';
y_qam16 = awgn(y_qam16, 20);
y_qam162 = y_qam16.^2;
y_qam164 = y_qam16.^4;
y_qam168 = y_qam16.^8;
y_qam16_fft = fftshift(fft(y_qam16));
y_qam16_fft2 = fftshift(fft(y_qam162));
y_qam16_fft4 = fftshift(fft(y_qam164));
y_qam16_fft8 = fftshift(fft(y_qam168));
qam16_freqs = (-Fs / 2:Fs/length(y_qam16_fft):Fs / 2);

%%% PLOTS %%%
    %%% FSK %%%
scatterplot(y_fsk)
title("FSK Constellation")

figure()
plot(fsk_freqs(1:length(y_fsk_fft)), abs(y_fsk_fft))
title("FSK Orginal Spectrum")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

    %%% BPSK %%%
scatterplot(y_bpsk)
title("BPSK Constellation")

figure()
plot(bpsk_freqs(1:length(y_bpsk_fft)), abs(y_bpsk_fft))
title("BPSK Orginal Spectrum")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(bpsk_freqs(1:length(y_bpsk_fft2)), abs(y_bpsk_fft2))
title("BPSK Raised to Power Two")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(bpsk_freqs(1:length(y_bpsk_fft4)), abs(y_bpsk_fft4))
title("BPSK Raised to Power Four")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(bpsk_freqs(1:length(y_bpsk_fft8)), abs(y_bpsk_fft8))
title("BPSK Raised to Power Eight")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

    %%% QPSK %%%
scatterplot(y_qpsk)
title("QPSK Constellation")

figure()
plot(qpsk_freqs(1:length(y_qpsk_fft)), abs(y_qpsk_fft))
title("QPSK Orginal Spectrum")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qpsk_freqs(1:length(y_qpsk_fft2)), abs(y_qpsk_fft2))
title("QPSK Raised to Power Two")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qpsk_freqs(1:length(y_qpsk_fft4)), abs(y_qpsk_fft4))
title("QPSK Raised to Power Four")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qpsk_freqs(1:length(y_qpsk_fft8)), abs(y_qpsk_fft8))
title("QPSK Raised to Power Eight")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

    %%% 8PSK %%%
scatterplot(y_psk8)
title("8PSK Constellation")

figure()
plot(psk8_freqs(1:length(y_psk8_fft)), abs(y_psk8_fft))
title("8PSK Orginal Spectrum")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(psk8_freqs(1:length(y_psk8_fft2)), abs(y_psk8_fft2))
title("8PSK Raised to Power Two")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(psk8_freqs(1:length(y_psk8_fft4)), abs(y_psk8_fft4))
title("8PSK Raised to Power Four")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(psk8_freqs(1:length(y_psk8_fft8)), abs(y_psk8_fft8))
title("8PSK Raised to Power Eight")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

    %%% 8QAM %%%
scatterplot(y_qam8)
title("8QAM Constellation")

figure()
plot(qam8_freqs(1:length(y_qam8_fft)), abs(y_qam8_fft))
title("8QAM Orginal Spectrum")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qam8_freqs(1:length(y_qam8_fft2)), abs(y_qam8_fft2))
title("8QAM Raised to Power Two")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qam8_freqs(1:length(y_qam8_fft4)), abs(y_qam8_fft4))
title("8QAM Raised to Power Four")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qam8_freqs(1:length(y_qam8_fft8)), abs(y_qam8_fft8))
title("8QAM Raised to Power Eight")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

    %%% 16QAM %%%
scatterplot(y_qam16)
title("16QAM Constellation")

figure()
plot(qam16_freqs(1:length(y_qam16_fft)), abs(y_qam16_fft))
title("16QAM Orginal Spectrum")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qam16_freqs(1:length(y_qam16_fft2)), abs(y_qam16_fft2))
title("16QAM Raised to Power Two")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qam16_freqs(1:length(y_qam16_fft4)), abs(y_qam16_fft4))
title("16QAM Raised to Power Four")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")

figure()
plot(qam16_freqs(1:length(y_qam16_fft8)), abs(y_qam16_fft8))
title("16QAM Raised to Power Eight")
xlabel("Frequency (Hz)")
ylabel("Magnitude (dB)")