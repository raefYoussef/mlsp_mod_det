%% Env Setup
close all;
clear;

%% Params
datasetDir = "dataset";
classes = {'PSK-02', 'PSK-04', 'PSK-08', 'QAM-08', 'QAM-16', 'QAM-32', 'QAM-64'};

visualizeNetwork = false;

%% Generate Dataset (Optional if dataset exists)
numSamplesPerClass = 1000;
snrRange = [0 40];
phaseRotRange = [-pi/2, pi/2];
jitterStdRange = [0 .05];

if ~exist(datasetDir, 'dir')
    GenDataset(datasetDir, classes, numSamplesPerClass, snrRange, phaseRotRange, jitterStdRange);
end

%% Create the Datastores
imds = imageDatastore(datasetDir,"IncludeSubfolders",true, "LabelSource","foldernames");
imds.Labels = categorical(cellfun(@(x) regexp(x, strjoin(classes, '|'), 'match', 'once'), cellstr(imds.Files), 'UniformOutput', false));

% Define the regular expression patterns to extract SNR, phase rotation,
% and jitter
snrPattern = 'SNR_([0-9.]+)';
jitterPattern = 'Jitter_([0-9.]+)';
phasePattern = 'Phase_([-0-9.]+)';

% Initialize an array to store SNR values
numSamp = numel(imds.Files);
snrValues = NaN(numSamp, 1);
jitterValues = NaN(numSamp, 1);
phaseValues = NaN(numSamp, 1);

% Extract the metadata from the filepaths
for i = 1:numSamp
    filePath = imds.Files{i};
    
    % SNR 
    snrMatch = regexp(filePath, snrPattern, 'tokens', 'once');
    
    if ~isempty(snrMatch)
        snrValues(i) = str2double(snrMatch{1});
    end

    % Jitter
    jitterMatch = regexp(filePath, jitterPattern, 'tokens', 'once');
    
    if ~isempty(jitterMatch)
        jitterValues(i) = str2double(jitterMatch{1});
    end

    % Phase
    phaseMatch = regexp(filePath, phasePattern, 'tokens', 'once');
    
    if ~isempty(jitterMatch)
        phaseValues(i) = str2double(phaseMatch{1});
    end
end

% Combine all metadata
metadataTable = table(imds.Files, imds.Labels, snrValues, jitterValues, phaseValues, 'VariableNames', {'FilePath', 'Label', 'SNR', 'Jitter', 'Phase'});

%% Split Dataset into Train/Validation
[imdsTrain, imdsValid] = splitEachLabel(imds, 0.8, 'randomized');

%% Define CNN Architecture
inputSize = [size(readimage(imdsTrain,1),1), size(readimage(imdsTrain,1),2), 1]; % Adjust based on image size

% CNN for classification
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(numel(unique(imds.Labels))) % Number of classes
    softmaxLayer
    classificationLayer];

% Display the network architecture
if visualizeNetwork
    analyzeNetwork(layers);
end

%% Train the Network
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 2, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValid, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(imdsTrain, layers, options);

%% Evaluate 
% Classify validation images 
predictedLabels = classify(net, imdsValid);
trueLabels = imdsValid.Labels;

% Calculate accuracy
accuracy = mean(predictedLabels == trueLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Create Confusion matrix
confMat = confusionmat(trueLabels, predictedLabels);
confusionchart(trueLabels, predictedLabels);

disp('Confusion Matrix:');
disp(confMat);
