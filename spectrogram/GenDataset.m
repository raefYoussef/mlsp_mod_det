function GenDataset(outputDir, classes, numSamplesPerClass, snrRange, phaseRotRange, jitterStdRange)
% Generates a dataset of modulated signals and saves constellation plots.
%
% Inputs:
%   outputDir          - Directory to save generated data
%   classes            - Cell array of modulation classes (e.g., {"PSK-02", "PSK-04", "QAM-16"})
%   numSamplesPerClass - Number of samples to generate for each modulation class
%   snrRange           - SNR range [minSNR maxSNR]
%   phaseRotRange      - Phase rotation range [minRot maxRot]
%   jitterStdRange     - Jitter standard deviation range [minJitter maxJitter]

    % Init RNG
    rng(1);
    
    % create dataset directory 
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Loop through each modulation class
    for i = 1:length(classes)
        modClass = classes{i};
        fprintf("Generating samples for %s\n", modClass);
        
        classDir = fullfile(outputDir, modClass);
        if ~exist(classDir, 'dir')
            mkdir(classDir);
        end

        % Generate samples until we reach the desired number per class
        generatedSamples = 0;
        while generatedSamples < numSamplesPerClass
            throughput = randsample([15 30],1);
            % Generate a signal using GenModSig
            [Fs, sigIQ, sigSym, sigClass, sigSNR, sigPhase, sigJitter] = ...
                GenModSig(true, snrRange, phaseRotRange, jitterStdRange, {modClass}, throughput); 

            % Define a unique directory for this sample that can house different features
            sampleDir = fullfile(classDir, sprintf('SNR_%.1f_Jitter_%.2f_Phase_%.2f', ...
                sigSNR, sigJitter, sigPhase));
            
            if ~exist(sampleDir, 'dir')
                mkdir(sampleDir);
            end

            % Save the IQ Data
            save(fullfile(sampleDir, 'signalIQ.mat'),'sigIQ');

            % Generate constellation
            constellation = GenConst(sigIQ);

            % Save the constellation plot as an image in the sample directory
            fileName = sprintf('constellation.png');
            imwrite(constellation, fullfile(sampleDir, fileName));

            % Compute the spectrogram
            figure;
            window_length = 50;
            nfft = 64;
            [spectrogramData, freq, time] = spectrogram(sigIQ, window_length, window_length/10, nfft, Fs);
            spectrogramData = 20*log10(abs(spectrogramData(:,1:200)));
            spectrogramImage = mat2gray(spectrogramData);
            imwrite(spectrogramImage, fullfile(sampleDir, 'spectrogram.png'));

            % Increment the count for successfully generated samples
            generatedSamples = generatedSamples + 1;
            close all
        end
    end

    fprintf("Dataset generation complete.\n");
end
