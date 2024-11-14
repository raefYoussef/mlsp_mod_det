%% Env Setup
close all;
clear;

%% Params
datasetFlag = true;

%% Generate Dataset (Optional if dataset exists)
outputDir = "dataset";
classes = {"2-PSK", "4-PSK", "8-PSK", "16-QAM", "32-QAM", "64-QAM"};
numSamplesPerClass = 1000;
snrRange = [20 40];
phaseRotRange = [-pi/2, pi/2];
jitterStdRange = [0 .05];

if datasetFlag
    GenDataset(outputDir, classes, numSamplesPerClass, snrRange, phaseRotRange, jitterStdRange);
end