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
imgDs = imageDatastore(datasetDir, "IncludeSubfolders", true, "LabelSource", "foldernames");

% Define regex patterns to extract SNR, phase, and jitter
snrPattern = 'SNR_([0-9.]+)';
jitterPattern = 'Jitter_([0-9.]+)';
phasePattern = 'Phase_([-0-9.]+)';

% Initialize metadata arrays
numSamp = numel(imgDs.Files);
snrValues = NaN(numSamp, 1);
jitterValues = NaN(numSamp, 1);
phaseValues = NaN(numSamp, 1);

% Extract metadata from filepaths
for i = 1:numSamp
    filePath = imgDs.Files{i};
    
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

% Combine all metadata into a table
metadataTable = table(imgDs.Files, snrValues, jitterValues, phaseValues, ...
    'VariableNames', {'FilePath', 'SNR', 'Jitter', 'Phase'});

%% Split Dataset into Train/Validation
numTrain = round(0.8 * numSamp);
randIndices = randperm(numSamp);

% Split images and corresponding labels
imgDsTrain = subset(imgDs, randIndices(1:numTrain));
imgDsValid = subset(imgDs, randIndices(numTrain+1:end));

trainTargets = [snrValues(randIndices(1:numTrain)), jitterValues(randIndices(1:numTrain)), phaseValues(randIndices(1:numTrain))];
validTargets = [snrValues(randIndices(numTrain+1:end)), jitterValues(randIndices(numTrain+1:end)), phaseValues(randIndices(numTrain+1:end))];

%% Define CNN Architecture
inputSize = [size(readimage(imgDsTrain,1),1), size(readimage(imgDsTrain,1),2), 1];

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
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(3) % Output layer for 3 targets (SNR, jitter, phase)
    regressionLayer]; % Regression layer for continuous output

% Visualize the network
if visualizeNetwork
    analyzeNetwork(layers);
end

%% Prepare Training Data
% Create augmented image datastores for training
% augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain);
% augmentedValid = augmentedImageDatastore(inputSize, imdsValid);

% Combine images and numerical targets for regression
trainTargets = arrayDatastore(trainTargets, 'IterationDimension', 1);
validTargets = arrayDatastore(validTargets, 'IterationDimension', 1);

trainDatastore = combine(imgDsTrain, trainTargets);
validDatastore = combine(imgDsValid, validTargets);

%% Train the Network
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validDatastore, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net = trainNetwork(trainDatastore, layers, options);

%% Evaluate
% Predict on validation set
YRegPred = predict(net, validDatastore).';
YRegTrue = cell2mat(readall(validTargets)).';

% Compute Mean Squared Error
mse = mean((YRegPred - YRegTrue).^2, 'all');
fprintf('Validation MSE: %.4f\n', mse);

% Plot regression results for each target
targetNames = {'SNR', 'Jitter', 'Phase'};
for i = 1:size(YRegPred, 1)
    figure;
    scatter(YRegTrue(i, :), YRegPred(i, :), 'b.');
    hold on;
    plot([min(YRegTrue(i, :)), max(YRegTrue(i, :))], ...
         [min(YRegTrue(i, :)), max(YRegTrue(i, :))], 'r--'); % Ideal fit line
    hold off;
    xlabel(sprintf('True %s', targetNames{i}));
    ylabel(sprintf('Predicted %s', targetNames{i}));
    title(sprintf('Regression Results for %s', targetNames{i}));
    grid on;
end
