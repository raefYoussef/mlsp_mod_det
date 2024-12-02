close all;
clear;
clc;

% Bool to generate dataset
datasetFlag = false;

% Generate Dataset (Optional if dataset exists)
% Set our output directory, classes, number of samples per class, snr
% range, phase rotation range, and jitter standard deviation range
outputDir = "dataset";
classes = {'PSK-02', 'PSK-04', 'PSK-08', 'QAM-08', 'QAM-16', 'QAM-32', 'QAM-64'};
numSamplesPerClass = 1000;
snrRange = [20 40];
phaseRotRange = [-pi/2, pi/2];
jitterStdRange = [0 .05];

% If we want to generate a dataset
if datasetFlag
    GenDataset(outputDir, classes, numSamplesPerClass, snrRange, phaseRotRange, jitterStdRange);
end

% Get overall accuracy and solution/guess result matrix
[accuracy, sol_gss_res] = KingCluster();

disp("Accuracy for modulation type was: " + string(accuracy * 100) + "%")

% Plot solution/guess matrix to confusion matrix
confusionchart(sol_gss_res(:, 1), sol_gss_res(:, 2))