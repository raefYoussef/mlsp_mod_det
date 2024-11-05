function [Fs, sigIQ, sigSym, sigClass, sigSNR, sigPhase, sigPhaseErrors] = GenModSig(Fs, throughput)
% Function to generate and augment baseband modulated signals (PSK, QAM)
%
% Inputs:
%   Fs              - Sampling frequency (Hz) (Multiple of 60)
%   throughput      - Desired throughput (bits per second)(Multiple of 60)
% Outputs:
%   Fs              - Actual Fs 
%   sigIQ           - Signal IQ
%   sigSym          - Signal Symbols
%   sigClass        - Signal Class - ["2-PSK", "4-PSK", "8-PSK", "16-QAM", "32-QAM", "64-QAM"]
%   sigSNR          - Signal SNR
%   sigPhase        - Signal Phase
%   sigPhaseErrors  - Array of applied phase errors for each symbol

    % Constants
    msgLen = 120; 
    modClasses = ["2-PSK", "4-PSK", "8-PSK", "16-QAM", "32-QAM", "64-QAM"];
    phaseRange = [0, pi/2];
    snrRange = [-30, 30];
    phaseErrorRange = [-pi/36, pi/36]; % Reduced phase error range (-5 degrees to +5 degrees)

    % Input Conditioning 
    Fs = 60*ceil(Fs/60);
    throughput = 60*ceil(throughput/60);

    % Randomly select signal parameters
    sigClass = modClasses(randi(length(modClasses)));
    sigSNR = snrRange(1) + (snrRange(2) - snrRange(1)) * rand;
    sigPhase = phaseRange(1) + (phaseRange(2) - phaseRange(1)) * rand;

    % Determine modulation order and bits per symbol
    switch sigClass
        case '2-PSK'
            M = 2; % Modulation order
            bitsPerSymbol = 1;
        case '4-PSK'
            M = 4;
            bitsPerSymbol = 2;
        case '8-PSK'
            M = 8;
            bitsPerSymbol = 3;
        case '16-QAM'
            M = 16;
            bitsPerSymbol = 4;
        case '32-QAM'
            M = 32;
            bitsPerSymbol = 5;
        case '64-QAM'
            M = 64;
            bitsPerSymbol = 6;
        otherwise
            error('Unsupported modulation class');
    end

    % Calculate required symbol rate to achieve the target throughput
    symRate = throughput / bitsPerSymbol; % Symbol rate (symbols per second)

    % Generate a random message symbols
    msg = randi([0 M-1], msgLen / bitsPerSymbol, 1);

    % Modulate the message based on selected class
    if contains(sigClass, 'PSK')
        sigSym = pskmod(msg, M, sigPhase); % PSK modulation
    else
        sigSym = qammod(msg, M); % QAM modulation
        % Apply phase offset to the QAM symbols
        sigSym = sigSym * exp(1j * sigPhase); % Rotate QAM symbols by sigPhase
    end

    % Generate variable phase errors for each symbol
    sigPhaseErrors = phaseErrorRange(1) + (phaseErrorRange(2) - phaseErrorRange(1)) * rand(length(msg), 1);

    % Apply phase errors to each symbol
    sigSym = sigSym .* exp(1j .* sigPhaseErrors); % Rotate by variable phase error

    % Pulse shaping using a raised cosine filter
    rollOff = 0.25; % Roll-off factor
    span = 6; % Filter span in symbols
    sps = Fs / symRate; % Samples per symbol
    rcosFilter = rcosdesign(rollOff, span, sps, 'sqrt'); % Raised cosine filter

    % Upsample the modulated symbols
    upsampledSigSym = upsample(sigSym, sps);

    % Filter the upsampled signal with the raised cosine filter
    sigIQ = conv(upsampledSigSym, rcosFilter, 'same');

    % Add Gaussian noise at specified SNR
    sigIQ = awgn(sigIQ, sigSNR, 'measured'); % Add noise to the signal
end
    