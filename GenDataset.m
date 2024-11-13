function GenDataset(outputDir, classes, numSamplesPerClass, snrRange, phaseRotRange, jitterStdRange)
% Generates a dataset of modulated signals and saves constellation plots.
%
% Inputs:
%   outputDir          - Directory to save generated data
%   classes            - Cell array of modulation classes (e.g., {"2-PSK", "4-PSK", "16-QAM"})
%   numSamplesPerClass - Number of samples to generate for each modulation class
%   snrRange           - SNR range [minSNR maxSNR]
%   phaseRotRange      - Phase rotation range [minRot maxRot]
%   jitterStdRange     - Jitter standard deviation range [minJitter maxJitter]

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
            % Generate a signal using GenModSig
            [Fs, sigIQ, sigSym, sigClass, sigSNR, sigPhase, sigJitter] = ...
                GenModSig(true, snrRange, phaseRotRange, jitterStdRange, {modClass}); 

            % Generate constellation
            const = GenConst(sigIQ);

            % Define a unique directory for this sample that can house different features
            sampleDir = fullfile(classDir, sprintf('SNR_%.1f_Jitter_%.2f_Phase_%.2f', ...
                sigSNR, sigJitter, sigPhase));
            
            if ~exist(sampleDir, 'dir')
                mkdir(sampleDir);
            else
                continue;
            end

            % Save the constellation plot as an image in the sample directory
            fileName = sprintf('constellation.png');
            imwrite(const, fullfile(sampleDir, fileName));
            
            % Increment the count for successfully generated samples
            generatedSamples = generatedSamples + 1;
        end
    end

    fprintf('Dataset generation complete.');
end
