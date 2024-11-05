clear;
close all;

% Examples
numEx = 10;

for exIndx = 1:numEx
    [Fs, sigIQ, sigSym, sigClass, sigSNR, sigPhase, sigJitter] = GenModSig(true, [25 40], [-pi/2, pi/2], [0 .05]); 
    const = GenConst(sigIQ); 
    
    figure(); 
    imagesc(const); 
    colorbar;
    title(sprintf("%s, SNR = %.2f dB, Rotation = %.2f rad, Jitter = %.2f", sigClass, sigSNR, sigPhase, sigJitter));
end