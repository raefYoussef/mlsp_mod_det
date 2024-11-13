function [Fs, sigIQ, sigSym, sigClass, sigSNR, sigPhase, sigJitter] = GenModSig(agument, snrRange, phaseRotRange, jitterStdRange, modSubset)
% Function to generate and augment baseband modulated signals (PSK, QAM, FSK)
%
% Inputs:
%   agument         - Flag to add data agumentation
%   snrRange        - SNR Range [minSNR maxSNR]
%   phaseRotRange   - Phase Rotation Range [minRotPhase maxRotPhase]
%   jitterStdRange  - Jitter Standard Dev Range [minJitter, maxJitter] 
%   modSubset       - Cell array of modulation classes to include (e.g., {"2-PSK", "4-PSK", "16-QAM"})
%
% Outputs:
%   Fs              - Actual Fs 
%   sigIQ           - Signal IQ
%   sigSym          - Signal Symbols
%   sigClass        - Signal Class
%   sigSNR          - Signal SNR
%   sigPhase        - Signal Phase
%   sigJitter       - Signal Jitter Std Dev

    % Constants
    Fs = 120;
    throughput = 60;
    msgLen = 120 * 100; 
    defaultModClasses = ["2-PSK", "4-PSK", "8-PSK", "16-QAM", "32-QAM", "64-QAM"];
    
    % Set modClasses to modSubset if provided, otherwise use default classes
    if nargin < 5 || isempty(modSubset)
        modClasses = defaultModClasses;
    else
        modClasses = intersect(defaultModClasses, string(modSubset)); % Ensure valid classes only
        if isempty(modClasses)
            error('No valid modulation classes specified in modSubset.');
        end
    end
    
    % Randomly select signal parameters
    sigClass = modClasses(randi(length(modClasses)));

    if agument
        sigSNR = snrRange(1) + (snrRange(2) - snrRange(1)) * rand;
        sigPhase = phaseRotRange(1) + (phaseRotRange(2) - phaseRotRange(1)) * rand;
    else
        sigSNR = 30;
        sigPhase = 0;
    end

    % Determine modulation order and bits per symbol
    switch sigClass
        case '2-PSK'
            M = 2;
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

    % Generate random message symbols
    msg = randi([0 M-1], msgLen / bitsPerSymbol, 1);

    % Modulate based on the selected class
    if contains(sigClass, 'PSK')
        sigSym = pskmod(msg, M, sigPhase); % PSK modulation
    elseif contains(sigClass, 'QAM')
        sigSym = qammod(msg, M); % QAM modulation
        sigSym = sigSym * exp(1j * sigPhase); % Rotate by phase
    end

    % Generate variable phase errors for each symbol
    if agument
        jitter = jitterStdRange(1) + (jitterStdRange(2) - jitterStdRange(1)) * rand;
        phaseJitter = jitter * randn(size(sigSym));  % Gaussian distributed phase error
        sigJitter = std(phaseJitter);
        sigSym = sigSym .* exp(1j .* phaseJitter);      % Rotate by variable phase error
    end

    % Pulse shaping using a raised cosine filter
    rollOff = 0.25; % Roll-off factor
    span = 2; % Filter span in symbols
    sps = Fs / symRate; % Samples per symbol
    rcosFilter = rcosdesign(rollOff, span, sps, 'sqrt'); % Raised cosine filter

    % Upsample the modulated symbols
    upsampledSigSym = upsample(sigSym, sps);

    % Filter the upsampled signal with the raised cosine filter
    sigIQ = conv(upsampledSigSym, rcosFilter, 'same');

    % Add Gaussian noise at specified SNR
    if agument
        sigIQ = awgn(sigIQ, sigSNR, 'measured'); % Add noise to the signal
    end
end
    