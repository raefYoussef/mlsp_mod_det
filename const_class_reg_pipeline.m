%% Env Setup
close all;
clear;

%% Params
datasetDir = "dataset";
classes = {'PSK-02', 'PSK-04', 'PSK-08', 'QAM-08', 'QAM-16', 'QAM-32', 'QAM-64'};

visualizeNetwork = false;

%% Generate Dataset
numSamplesPerClass = 1000;
snrRange = [0 40];
phaseRotRange = [-pi/2, pi/2];
jitterStdRange = [0 .05];

if ~exist(datasetDir, 'dir')
    GenDataset(datasetDir, classes, numSamplesPerClass, snrRange, phaseRotRange, jitterStdRange);
end

%% Create the Datastores
imgDs = imageDatastore(datasetDir,"IncludeSubfolders",true, "LabelSource","foldernames");
imgDs.Labels = categorical(cellfun(@(x) regexp(x, strjoin(classes, '|'), 'match', 'once'), cellstr(imgDs.Files), 'UniformOutput', false));

% Define the regular expression patterns to extract SNR, phase rotation,
% and jitter
snrPattern = 'SNR_([0-9.]+)';
jitterPattern = 'Jitter_([0-9.]+)';
phasePattern = 'Phase_([-0-9.]+)';

% Initialize an array to store SNR values
numSamp = numel(imgDs.Files);
snrValues = NaN(numSamp, 1);
jitterValues = NaN(numSamp, 1);
phaseValues = NaN(numSamp, 1);

% Extract the metadata from the filepaths
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
    if ~isempty(phaseMatch)
        phaseValues(i) = str2double(phaseMatch{1});
    end
end

% Combine all metadata
metadataTable = table(imgDs.Files, onehotencode(imgDs.Labels, 2), snrValues, jitterValues, phaseValues, 'VariableNames', {'FilePath', 'Label', 'SNR', 'Jitter', 'Phase'});

%% Combined Datastore
trainSplit = .8;

numTrain = round(trainSplit * numSamp);
randIndices = randperm(numSamp);
trainIndx = randIndices(1:numTrain);
validIndx = randIndices(numTrain+1:end);

% Split the metadata based on the indices
imgDsTrain = subset(imgDs, trainIndx);
imgDsValid = subset(imgDs, validIndx);

metadataTrain = metadataTable(trainIndx, :);
metadataValid = metadataTable(validIndx, :);

% Create Label Datastore (outputs categorical labels)
labelDsTrain = arrayDatastore(metadataTrain{:, {'Label'}});
labelDsValid = arrayDatastore(metadataValid{:, {'Label'}});

% Create Regression Datastore (outputs SNR, Jitter, and Phase)
regDsTrain = arrayDatastore(metadataTrain{:, {'SNR', 'Jitter', 'Phase'}});
regDsValid = arrayDatastore(metadataValid{:, {'SNR', 'Jitter', 'Phase'}});

% Combine imageDatastore and metadata datastore
combDsTrain = combine(imgDsTrain, labelDsTrain, regDsTrain);
combDsValid = combine(imgDsValid, labelDsValid, regDsValid);

%% Define CNN Architecture
imgSize = [size(readimage(imgDsTrain,1),1), size(readimage(imgDsTrain,1),2), 1]; % Adjust based on image size
numClasses = numel(unique(imgDs.Labels));

% Two-headed CNN architecture for classification and regression
backboneLayers = [
    imageInputLayer(imgSize, 'Name', 'input')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    
    % Shared feature extraction layers end here
    fullyConnectedLayer(128, 'Name', 'fc_shared')
    reluLayer('Name', 'relu_shared')
];

% Classification branch
classHead = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_classification')
    softmaxLayer('Name', 'softmax')
];

% Regression branch
regHead = [
    fullyConnectedLayer(64, 'Name', 'fc_regression')
    reluLayer('Name', 'relu_regression')
    fullyConnectedLayer(3, 'Name', 'fc_regression_output') % 3 regression targets (SNR, Jitter, Phase)
];

% Create layer graph from the backbone
lgraph = layerGraph(backboneLayers);

% Add classification head
lgraph = addLayers(lgraph, classHead);
lgraph = connectLayers(lgraph, 'relu_shared', 'fc_classification');

% Add regression head
lgraph = addLayers(lgraph, regHead);
lgraph = connectLayers(lgraph, 'relu_shared', 'fc_regression');

% Display the network architecture
if visualizeNetwork
    analyzeNetwork(lgraph);
end

%% Train the Network
% Initialize the network
net = dlnetwork(lgraph); 

% Training parameters
learnRate = 0.001;              % Initial learning rate
gradientDecay = 0.9;            % Gradient decay rate for Adam
squaredGradientDecay = 0.999;   % Squared gradient decay rate for Adam

numEpochs = 10;
miniBatchSize = 32;
valFreq = 10;

% Process minibatch for training
mbq = minibatchqueue(combDsTrain, ...
    'MiniBatchSize', miniBatchSize, ...
    'MiniBatchFcn', @(images, labels, regTarget) preprocessMiniBatch(images, labels, regTarget), ...
    'MiniBatchFormat', {'SSCB', '', ''}, ...            % Image data, labels, and regression targets
    'OutputAsDlarray', [true, true, true], ...          % Convert all to dlarray for GPU
    'OutputEnvironment', ["auto", "auto", "auto"]);     % Ensure compatibility with GPU

% Training loop
averageGrad = [];
averageSqGrad = [];

% Initialize the training progress monitor
monitor = trainingProgressMonitor(Metrics=["TrainLoss", "ValidLoss", ...
    "TrainClassLoss", "ValidClassLoss", "TrainRegLoss", "ValidRegLoss"], ...
    Info="Epoch", XLabel="Iteration");

% Overlay related metrics on the same subplot
groupSubPlot(monitor, "Loss", ["TrainLoss", "ValidLoss"]);
groupSubPlot(monitor, "Classification Loss", ["TrainClassLoss", "ValidClassLoss"]);
groupSubPlot(monitor, "Regression Loss", ["TrainRegLoss", "ValidRegLoss"]);

numSamp = numel(imgDsTrain.Files);
numIterPerEpoch = floor(numSamp / miniBatchSize);
numIter = numEpochs * numIterPerEpoch;
iter = 0;
epoch = 0;

while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle the mini-batch queue
    shuffle(mbq);

    % Initialize cumulative loss for the epoch
    totalEpochLoss = 0;

    i = 0;
    while i < numIterPerEpoch && ~monitor.Stop
        i = i + 1;
        iter = iter + 1;
        
        % Read a mini-batch
        [X, YClassification, YRegression] = next(mbq);

        % Perform forward and backward passes using dlfeval
        [trainLoss, gradients, trainClassLoss, trainRegLoss, state] = dlfeval(@modelGradients, net, X, YClassification, YRegression);
        net.State = state;

        % Update the network parameters using the Adam optimizer.
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iter, learnRate, gradientDecay, squaredGradientDecay);
        
        % Update the total loss for tracking
        totalEpochLoss = totalEpochLoss + double(trainLoss);

        % Perform validation at the specified frequency
        if mod(iter, valFreq) == 0
            [YClassTrue, YClassPred, YRegTrue, YRegPred] = performValidation(net, combDsValid);

            % Compute Losses
            validClassLoss = crossentropy(YClassPred, YClassTrue);
            validRegLoss = mean((YRegPred - YRegTrue).^2, 'all');
            validLoss = validClassLoss + validRegLoss;
        
            % Record validation metrics in the monitor
            recordMetrics(monitor,iter, ValidLoss=validLoss, ValidClassLoss=validClassLoss, ValidRegLoss=validRegLoss);
        else
            
        end

        % Update the training progress monitor.
        recordMetrics(monitor,iter, TrainLoss=trainLoss, TrainClassLoss=trainClassLoss, TrainRegLoss=trainRegLoss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iter/numIter;
    end

    % Display average loss for the epoch
    avgLoss = totalEpochLoss / numIterPerEpoch;
    fprintf('Epoch %d/%d: Avg Loss = %.4f\n', epoch, numEpochs, avgLoss);
end

%% Evaluate Train Dataset
[YClassTrue, YClassPred, YRegTrue, YRegPred] = performValidation(net, combDsValid);

%% Eval Classification
% Convert predicted probabilities to class labels
[~, classIdx] = max(YClassPred, [], 1);
YClassPred = categorical(classes(classIdx));

% Get true class labels
YClassTrue = (imgDsValid.Labels).';

% Calculate accuracy
accuracy = mean(YClassPred == YClassTrue);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Confusion Matrix
confMat = confusionmat(YClassTrue, YClassPred);
confusionchart(YClassTrue, YClassPred, 'Title', 'Confusion Matrix');

%% Eval Regression
% Compute Mean Squared Error (MSE)
mseValue = mean((YRegPred - YRegTrue).^2, 'all');
fprintf('Validation MSE: %.4f\n', mseValue);

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

%% Helper Functions
function [X, YClassification, YRegression] = preprocessMiniBatch(images, labels, regTarget)
    % Convert images to a 4D array
    concatImages = [];
    for i = 1:numel(images)
        img = images{i};
        concatImages = cat(4, concatImages, img);
    end
    
    % Single is friendly to DNNs
    X = single(concatImages); 
    
    % Unpack labels from cell and convert to double
    YClassification = cell2mat(labels).';

    % Unpack regression targets from cell
    YRegression = cell2mat(regTarget).';
end


function [loss, gradients, classLoss, regLoss, state] = modelGradients(net, X, YClassification, YRegression)
    % Forward pass through the network
    [YPredClass, YPredReg, state] = forward(net, X);

    % Compute classification loss (cross-entropy)
    classLoss = crossentropy(YPredClass, YClassification);

    % Compute regression loss (mean squared error)
    regLoss = mean((YPredReg - YRegression).^2, 'all');

    % Total loss (weighted sum of classification and regression losses)
    loss = classLoss + regLoss;

    % Compute gradients (backpropagation)
    gradients = dlgradient(loss, net.Learnables);
end

function [YClassTrue, YClassPred, YRegTrue, YRegPred] = performValidation(net, combDsValid)
    numSamp = combDsValid.numpartitions;
    sampPrev = preview(combDsValid);
    numClasses = size(cell2mat(sampPrev(2)), 2);
    numRegVars = size(cell2mat(sampPrev(3)), 2);

    YClassPred = zeros(numClasses, numSamp);
    YClassTrue = zeros(numClasses, numSamp);

    YRegPred = zeros(numRegVars, numSamp);
    YRegTrue = zeros(numRegVars, numSamp);

    % Evaluate Classification
    reset(combDsValid); 
    i = 1;
    while hasdata(combDsValid)
        batch = read(combDsValid);
        % pre-process batch
        img = dlarray(single(batch{1}), 'SSCB');
        YClassTrue(:, i) = batch{2}.';
        YRegTrue(:, i) = batch{3}.';
        % forward pass
        [YClassPred(:, i), YRegPred(:, i)] = predict(net, img); 
        i = i+1;
    end
end

