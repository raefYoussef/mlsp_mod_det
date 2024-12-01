close all
clear all

%% Define Classes
classes = {'PSK-02', 'PSK-04', 'PSK-08', 'QAM-08', 'QAM-16', 'QAM-32', 'QAM-64', 'FSK-02', 'FSK-04', 'FSK-08'};
%classes = {'PSK-08', 'QAM-08'};
visualizeNetwork = false;
%% Generate Dataset (Optional if dataset exists)
numSamplesPerClass = 1000;
snrRange = [10 30];
phaseRotRange = [-pi/2, pi/2];
jitterStdRange = [0 .05];
datasetDir = "dataset";
%GenDataset(datasetDir, classes, numSamplesPerClass, snrRange, phaseRotRange, jitterStdRange);

%% Create a datastore
datasetDir = "dataset";
imds = imageDatastore(datasetDir,"IncludeSubfolders",true,"LabelSource","foldernames");
allFiles = imds.Files;
%selectedFile = "spectrogram.png";
selectedFile = "constellation.png";
selection = allFiles(contains(allFiles, selectedFile));
imds = imageDatastore(selection);
imds.Labels = categorical(cellfun(@(x) regexp(x, strjoin(classes, '|'), 'match', 'once'), cellstr(imds.Files), 'UniformOutput', false));

%% Split into Training and Validation Sets
[imdsTrain, imdsValid] = splitEachLabel(imds, 0.8, 'randomized');

%% Define CNN Architecture
inputSize = [size(readimage(imdsTrain,1),1), size(readimage(imdsTrain,1),2), 1]; % Adjust based on image size

%% CNN for classification
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

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
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
    'MaxEpochs', 3, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValid, ...
    'ValidationFrequency', 20, ...
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
