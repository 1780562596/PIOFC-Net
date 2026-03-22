# PIOFC-Net
A deblurring model based on optical flow networks
%% PIOFC-Net
clc; clear; close all;
global featureLayer;
featureLayer = 'relu5_4';
customLayerNames = {'channelAttentionLayer'}; 
for i = 1:numel(customLayerNames)
    if isempty(which(customLayerNames{i}))
        customLayerPath = fileparts(mfilename('fullpath')); 
        addpath(fullfile(customLayerPath, 'custom_layers'));
        if isempty(which(customLayerNames{i}))
            error('自定义层 %s 未找到。请确保文件存在', customLayerNames{i});
        end
    end
end
%% Data preprocessing
datasetRoot1 = 'F:\zhuomian\GOPRO\train\input';
datasetRoot2 = 'F:\zhuomian\GOPRO\train\target';
datasetRoot3 = 'F:\zhuomian\GOPRO\test\blur';
datasetRoot4 = 'F:\zhuomian\GOPRO\test\sharp';
imds1 = imageDatastore(datasetRoot1, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds2 = imageDatastore(datasetRoot2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds3 = imageDatastore(datasetRoot3, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds4 = imageDatastore(datasetRoot4, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% ==================== Ablation experiment configuration =====================
ablationConfig = struct();
ablationConfig.useAttention = true;      
ablationConfig.numResidualBlocks = 0;   
ablationConfig.useMultiScaleD = true;  
ablationConfig.useFlowNet = true;      
ablationConfig.usePerceptualLoss = true; 
ablationConfig.trainingPhases = [1,2,3]; 
resultDir = fullfile(pwd, 'paper_results'); 
if ~exist(resultDir, 'dir')
    mkdir(resultDir); 
end
save(fullfile(resultDir, 'ablation_config.mat'), 'ablationConfig');
%% Dataset splitting
fprintf('Number of blurry images in the training set: %d\n', numel(imds1.Files));
fprintf('Number of clear images in the training set: %d\n', numel(imds2.Files));
fprintf('Number of blurry images in the test set: %d\n', numel(imds3.Files));
fprintf('Number of clear images in the test set: %d\n', numel(imds4.Files));
if numel(imds1.Files) ~= numel(imds2.Files)
    error('error');
end
if numel(imds3.Files) ~= numel(imds4.Files)
    error('error');
end
trainImages = numel(imds1.Files);
trainRatio = 0.85;
valRatio = 0.15;
allIndices = randperm(trainImages);
trainEnd = floor(trainImages * trainRatio);
valEnd = trainEnd + floor(trainImages * valRatio);
trainIdx = allIndices(1:trainEnd);
valIdx = allIndices(trainEnd+1:valEnd);
testIdx = allIndices(valEnd+1:end);
imdsBlurTrain = subset(imds1, trainIdx);
imdsSharpTrain = subset(imds2, trainIdx);
imdsBlurVal = subset(imds1, valIdx);
imdsSharpVal = subset(imds2, valIdx);
imdsBlurTest = imds3;
imdsSharpTest = imds4;
fprintf('Dataset partitioning completed:\n');
fprintf('  training set: %d Image\n', numel(imdsBlurTrain.Files));
fprintf('  validation set: %d Image\n', numel(imdsBlurVal.Files));
fprintf('  test set: %d Image\n', numel(imdsBlurTest.Files));

%% ==================== Network architecture====================
inputSize = [256, 256, 3];
lgraph = layerGraph();
vggNet = loadVGG19FromMat();
featureLayer = 'relu5_4'; 
encoder = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    convolution2dLayer(7, 256, 'Padding', 'same', 'Stride', 1, 'Name', 'conv1')
    leakyReluLayer(0.2, 'Name', 'lrelu1')
    convolution2dLayer(3, 512, 'Padding', 'same', 'Stride', 2, 'Name', 'conv2')
    leakyReluLayer(0.2, 'Name', 'lrelu2')
    convolution2dLayer(3, 1024, 'Padding', 'same', 'Stride', 2, 'Name', 'conv3')
    leakyReluLayer(0.2, 'Name', 'lrelu3')
    convolution2dLayer(3, 1024, 'Padding', 'same', 'Stride', 2, 'Name', 'conv4')
    leakyReluLayer(0.2, 'Name', 'lrelu4')
];
lgraph = addLayers(lgraph, encoder);
lastLayer = 'lrelu4';
numResidualBlocks = ablationConfig.numResidualBlocks; 
for i = 1:numResidualBlocks
    blockPrefix = ['res_block' num2str(i)];
    if i == 1
        skipConv = convolution2dLayer(1, 512, 'Stride', 1, 'Padding', 'same', ...
            'Name', [blockPrefix, '_skip_conv']);
        lgraph = addLayers(lgraph, skipConv);
        lgraph = safeConnect(lgraph, lastLayer, [blockPrefix, '_skip_conv']);
        skipSource = [blockPrefix, '_skip_conv']; 
    else
        skipSource = lastLayer; 
    end
    conv1 = convolution2dLayer(3, 512, 'Padding', 'same', 'Name', [blockPrefix, '_conv1']);
    in1 = instanceNormalizationLayer('Name', [blockPrefix, '_in1']);
    relu1 = reluLayer('Name', [blockPrefix, '_relu1']);
    conv2 = convolution2dLayer(3, 512, 'Padding', 'same', 'Name', [blockPrefix, '_conv2']);
    in2 = instanceNormalizationLayer('Name', [blockPrefix, '_in2']);
    add = additionLayer(2, 'Name', [blockPrefix, '_add']);
    relu_out = reluLayer('Name', [blockPrefix, '_relu_out']);
    lgraph = addLayers(lgraph, conv1);
    lgraph = addLayers(lgraph, in1);
    lgraph = addLayers(lgraph, relu1);
    lgraph = addLayers(lgraph, conv2);
    lgraph = addLayers(lgraph, in2);
    lgraph = addLayers(lgraph, add);
    lgraph = addLayers(lgraph, relu_out);
    lgraph = safeConnect(lgraph, [blockPrefix, '_conv1'], [blockPrefix, '_in1']);
    lgraph = safeConnect(lgraph, [blockPrefix, '_in1'], [blockPrefix, '_relu1']);
    lgraph = safeConnect(lgraph, [blockPrefix, '_relu1'], [blockPrefix, '_conv2']);
    lgraph = safeConnect(lgraph, [blockPrefix, '_conv2'], [blockPrefix, '_in2']);
    lgraph = safeConnect(lgraph, [blockPrefix, '_in2'], [blockPrefix, '_add/in1']);
    lgraph = safeConnect(lgraph, skipSource, [blockPrefix, '_conv1']);
    lgraph = safeConnect(lgraph, skipSource, [blockPrefix, '_add/in2']);
    lgraph = safeConnect(lgraph, [blockPrefix, '_add'], [blockPrefix, '_relu_out']);
    lastLayer = [blockPrefix, '_relu_out'];
end
decoderLayers = [
    transposedConv2dLayer(3, 1024, 'Cropping', 'same', 'Stride', 2, 'Name', 'tconv1')
    leakyReluLayer(0.2, 'Name', 't_lrelu1')
];
lgraph = addLayers(lgraph, decoderLayers);
lgraph = safeConnect(lgraph, lastLayer, 'tconv1');
decoderStages = {
    struct('input', 't_lrelu1', 'skip', 'lrelu3', 'attn', 'attn1', 'concat', 'concat1', 'output', 'decoder2', 'outChannels', 512)
    struct('input', 'decoder2_out', 'skip', 'lrelu2', 'attn', 'attn2', 'concat', 'concat2', 'output', 'decoder3', 'outChannels', 256)
    struct('input', 'decoder3_out', 'skip', 'lrelu1', 'attn', 'attn3', 'concat', 'concat3', 'output', 'final_conv', 'outChannels', 3)
};
prevStageOutput = 't_lrelu1';
  
for i = 1:numel(decoderStages)
    stage = decoderStages{i};
    concatName = [stage.attn '_concat']; 
    concatLayer = concatenationLayer(3, 2, 'Name', concatName);
    lgraph = addLayers(lgraph, concatLayer);
    if ablationConfig.useAttention
        attnLayer = channelAttentionLayer([stage.attn '_ca'], 16);
        lgraph = addLayers(lgraph, attnLayer);
        lgraph = safeConnect(lgraph, prevStageOutput, attnLayer.Name);
        lgraph = safeConnect(lgraph, attnLayer.Name, [concatName '/in1']);
    else
        lgraph = safeConnect(lgraph, prevStageOutput, [concatName '/in1']);
    end
    lgraph = safeConnect(lgraph, stage.skip, [concatName '/in2']);
    if i < numel(decoderStages)
        convLayer = transposedConv2dLayer(3, stage.outChannels, 'Cropping', 'same', 'Stride', 2, 'Name', stage.output);
    else
        convLayer = convolution2dLayer(7, stage.outChannels, 'Padding', 'same', 'Name', stage.output);
    end
    lgraph = addLayers(lgraph, convLayer);
    lgraph = safeConnect(lgraph, concatName, stage.output);
    if i < numel(decoderStages)
        actLayerName = [stage.output '_out'];
        actLayer = leakyReluLayer(0.2, 'Name', actLayerName);
        lgraph = addLayers(lgraph, actLayer);
        lgraph = safeConnect(lgraph, stage.output, actLayerName);
        prevStageOutput = actLayerName;
    else
        tanhLayer = tanhLayer('Name', 'output_tanh');
        lgraph = addLayers(lgraph, tanhLayer);
        lgraph = safeConnect(lgraph, stage.output, 'output_tanh');
        prevStageOutput = 'output_tanh';
    end
end
netG = dlnetwork(lgraph);
flowNet = buildOpticalFlowNet();
function netF = buildOpticalFlowNet()
input = imageInputLayer([256,256,6], 'Name','input');
conv1 = convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv1');
lrelu1 = leakyReluLayer(0.2,'Name','lrelu1');
conv2 = convolution2dLayer(3,64,'Padding','same','Stride',2,'Name','conv2');
lrelu2 = leakyReluLayer(0.2,'Name','lrelu2');
conv3 = convolution2dLayer(3,128,'Padding','same','Stride',2,'Name','conv3');
lrelu3 = leakyReluLayer(0.2,'Name','lrelu3');
tconv1 = transposedConv2dLayer(4,128,'Cropping','same','Stride',2,'Name','tconv1');
t_lrelu1 = leakyReluLayer(0.2,'Name','t_lrelu1');
concat1 = concatenationLayer(3,2,'Name','concat1');
tconv2 = transposedConv2dLayer(4,64,'Cropping','same','Stride',2,'Name','tconv2');
t_lrelu2 = leakyReluLayer(0.2,'Name','t_lrelu2');
concat2 = concatenationLayer(3,2,'Name','concat2');
tconv3 = transposedConv2dLayer(4,32,'Cropping','same','Stride',2,'Name','tconv3');
t_lrelu3 = leakyReluLayer(0.2,'Name','t_lrelu3');
concat3 = concatenationLayer(3,2,'Name','concat3');
flow_out = convolution2dLayer(3,2,'Padding','same','Name','flow_out');
lgraph = layerGraph();
layers = {input, conv1, lrelu1, conv2, lrelu2, conv3, lrelu3, ...
    tconv1, t_lrelu1, concat1, tconv2, t_lrelu2, concat2, ...
    tconv3, t_lrelu3, concat3, flow_out};

for i = 1:numel(layers)
    lgraph = addLayers(lgraph, layers{i});
end
lgraph = safeConnect(lgraph, 'input', 'conv1');
lgraph = safeConnect(lgraph, 'conv1', 'lrelu1');
lgraph = safeConnect(lgraph, 'lrelu1', 'conv2');
lgraph = safeConnect(lgraph, 'conv2', 'lrelu2');
lgraph = safeConnect(lgraph, 'lrelu2', 'conv3');
lgraph = safeConnect(lgraph, 'conv3', 'lrelu3');
lgraph = safeConnect(lgraph, 'lrelu3', 'tconv1');
lgraph = safeConnect(lgraph, 'tconv1', 't_lrelu1');
lgraph = safeConnect(lgraph, 't_lrelu1', 'concat1/in1');
lgraph = safeConnect(lgraph, 'lrelu2', 'concat1/in2');  
lgraph = safeConnect(lgraph, 'concat1', 'tconv2');
lgraph = safeConnect(lgraph, 'tconv2', 't_lrelu2');
lgraph = safeConnect(lgraph, 't_lrelu2', 'concat2/in1');
lgraph = safeConnect(lgraph, 'lrelu1', 'concat2/in2');  
lgraph = safeConnect(lgraph, 'concat2', 'tconv3');
lgraph = safeConnect(lgraph, 'tconv3', 't_lrelu3');
lgraph = safeConnect(lgraph, 't_lrelu3', 'concat3/in1');
lgraph = safeConnect(lgraph, 'input', 'concat3/in2');
lgraph = safeConnect(lgraph, 'concat3', 'flow_out');
netF = dlnetwork(lgraph);
end
%% Discriminator Network
function netD = buildDiscriminatorNet(inputSize)
    layers = [
        imageInputLayer([inputSize, inputSize, 6], 'Name', 'input')
        convolution2dLayer(4, 64, 'Stride', 2, 'Padding', 1, 'Name', 'conv1')
        leakyReluLayer(0.2, 'Name', 'lrelu1')
        convolution2dLayer(4, 128, 'Stride', 2, 'Padding', 1, 'Name', 'conv2')
        instanceNormalizationLayer('Name', 'in2') 
        leakyReluLayer(0.2, 'Name', 'lrelu2')
        convolution2dLayer(4, 256, 'Stride', 2, 'Padding', 1, 'Name', 'conv3')
        instanceNormalizationLayer('Name', 'in3') 
        leakyReluLayer(0.2, 'Name', 'lrelu3')
        convolution2dLayer(4, 512, 'Stride', 2, 'Padding', 1, 'Name', 'conv4')
        instanceNormalizationLayer('Name', 'in4') 
        leakyReluLayer(0.2, 'Name', 'lrelu4')
        convolution2dLayer(4, 1, 'Stride', 1, 'Padding', 1, 'Name', 'final_conv')
    ];
    lgraph = layerGraph(layers);
    netD = dlnetwork(lgraph);
end
if ablationConfig.useMultiScaleD
    netD1 = buildDiscriminatorNet(256);
    netD2 = buildDiscriminatorNet(128); 
    netD3 = buildDiscriminatorNet(64);  
else
    netD1 = buildDiscriminatorNet(256);
    netD2 = [];
    netD3 = [];
end
%% ==================== Loss function ====================
function [lossG, gradientsG] = modelLossG_phase1(netG, vggNet, XBlur, XSharp, featureLayer)
    XGenerated = forward(netG, XBlur);
    percep_loss = safePercepLoss(vggNet, XGenerated, XSharp, featureLayer);
    l1_loss = 15.0 * mean(abs(XGenerated - XSharp), 'all');
    ssim_val = mean(multissim(extractdata(XGenerated), extractdata(XSharp)));
    ssim_loss = 1.5 * (1 - ssim_val);
    grad_loss = gradientLoss(XGenerated, XSharp);
    lossG = l1_loss + 2.0 * percep_loss + ssim_loss + 0.5 * grad_loss;
    lossG = mean(lossG);
    gradientsG = dlgradient(lossG, netG.Learnables);
end
%% Gradient loss function
function loss = gradientLoss(img1, img2)
    img1_data = extractdata(img1);
    img2_data = extractdata(img2);
    if size(img1_data, 3) == 3
        gray1 = 0.2989 * img1_data(:,:,1,:) + ...
                0.5870 * img1_data(:,:,2,:) + ...
                0.1140 * img1_data(:,:,3,:);
        gray2 = 0.2989 * img2_data(:,:,1,:) + ...
                0.5870 * img2_data(:,:,2,:) + ...
                0.1140 * img2_data(:,:,3,:);
    else
        gray1 = img1_data;
        gray2 = img2_data;
    end
    total_loss = 0;
    num_images = size(gray1, 4);
    for i = 1:num_images
        img1_gray = gray1(:,:,:,i);
        img2_gray = gray2(:,:,:,i);
        [gx1, gy1] = imgradientxy(img1_gray);
        [gx2, gy2] = imgradientxy(img2_gray);
        gx_diff = mean(abs(gx1 - gx2), 'all');
        gy_diff = mean(abs(gy1 - gy2), 'all');
        total_loss = total_loss + gx_diff + gy_diff;
    end
    loss = total_loss / num_images;
end
%% ========== Paper Indicator Recording System ==========
metrics = struct();
metrics.train = struct('iteration', [], 'lossG', [], 'lossD', [], 'lossF', [], 'psnr', [], 'ssim', [], 'lpips', [], 'phase', []);
metrics.val = struct('iteration', [], 'psnr', [], 'ssim', [], 'lpips', [], 'time', []);
metrics.test = struct('psnr', [], 'ssim', [], 'lpips', [], 'fid', []);
resultDir = fullfile(pwd, 'paper_results');
if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end
metricTable = table('Size', [0, 8], ...
    'VariableNames', {'Iteration', 'Phase', 'LossG', 'LossD', 'LossF', 'PSNR', 'SSIM', 'LPIPS'}, ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'});
imageDir = fullfile(resultDir, 'comparison_images');
if ~exist(imageDir, 'dir')
    mkdir(imageDir);
end
%% ==================== Training strategy ====================
miniBatchSize = 1;
numIterationsG = 240000;
validationFrequency = 500;
numValidationImages = 4;
lossData = struct();
lossData.iteration = [];
lossData.lossG = [];
lossData.lossD = [];
lossData.lossF = [];
lossData.epoch = [];
lossCsvPath = fullfile(resultDir, 'training_losses.csv');
if ~exist(lossCsvPath, 'file')
    fid = fopen(lossCsvPath, 'w');
    fprintf(fid, 'iteration,epoch,lossG,lossD,lossF\n');
    fclose(fid);
end
maxMemoryEntries = 1000; 
initialLearnRateG = 2e-4;
% initialLearnRateD = 5e-5;
initialLearnRateD = 1e-4;
initialLearnRateF = 2e-5;
lrSchedule = @(iter, init, totalIter) init * (0.5*(1 + cos(pi*iter/totalIter)));
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.9;
mbqBlur = minibatchqueue(imdsBlurTrain, ...
    'MiniBatchSize', miniBatchSize, ...
    'PartialMiniBatch', 'discard', ...
    'MiniBatchFcn', @preprocessSingleImage, ...
    'MiniBatchFormat', 'SSCB');
mbqSharp = minibatchqueue(imdsSharpTrain, ...
    'MiniBatchSize', miniBatchSize, ...
    'PartialMiniBatch', 'discard', ...
    'MiniBatchFcn', @preprocessSingleImage, ...
    'MiniBatchFormat', 'SSCB');
imdsBlurTestRaw = imageDatastore(datasetRoot3, 'IncludeSubfolders', true);
validationData = [];
numValidationImages = 4; 
if numel(imdsBlurTestRaw.Files) < numValidationImages
    numValidationImages = numel(imdsBlurTestRaw.Files);
    warning('Insufficient test set images, only use %d Verify image', numValidationImages);
end
for i = 1:numValidationImages
    img = readimage(imdsBlurTestRaw, i);
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);
    end
    img = imresize(img, [256, 256]);
    if isa(img, 'uint8')
        img = single(img)/255.0;  % [0,255] -> [0,1]
    end
    img = (img * 2) - 1;
    validationData = cat(4, validationData, img);
end
validationData = dlarray(validationData, 'SSCB');
fprintf('Verify that the dataset creation is complete: %d  %dx%d image\n', ...
    numValidationImages, size(validationData,1), size(validationData,2));
[trailingAvgG, trailingAvgSqG] = deal([]);
[trailingAvgD1, trailingAvgSqD1] = deal([]);
[trailingAvgD2, trailingAvgSqD2] = deal([]);
[trailingAvgD3, trailingAvgSqD3] = deal([]);
[trailingAvgF, trailingAvgSqF] = deal([]);
lossD = dlarray(0);
lossG = dlarray(0);
lossF = dlarray(0);
figure;
subplot(2,2,[1,3]);
imageAxes = gca;
title('Generated Images');
subplot(2,2,2);
lineLossG = animatedline('Color', [0.85, 0.33, 0.1], 'LineWidth', 1.5);
lineLossD = animatedline('Color', [0, 0.45, 0.74], 'LineWidth', 1.5);
lineLossF = animatedline('Color', [0.49, 0.18, 0.56], 'LineWidth', 1.5);
legend('Generator Loss', 'Discriminator Loss', 'Flow Loss');
xlabel('Iteration');
ylabel('Loss');
title('Training Losses');
grid on;
subplot(2,2,4);
linePSNR = animatedline('Color', [0.47, 0.67, 0.19], 'LineWidth', 1.5);
lineSSIM = animatedline('Color', [0.85, 0.33, 0.1], 'LineWidth', 1.5);
lineLPIPS = animatedline('Color', [0.49, 0.18, 0.56], 'LineWidth', 1.5);
legend('PSNR', 'SSIM', 'LPIPS');
xlabel('Iteration');
ylabel('Metric Value');
title('Quality Metrics');
grid on;
infoPanel = uipanel('Position', [0.1, 0.01, 0.8, 0.05], 'Title', 'Training Status');
infoText = uicontrol('Style', 'text', 'Units', 'normalized', ...
    'Position', [0.02, 0.1, 0.96, 0.8], 'Parent', infoPanel, ...
    'HorizontalAlignment', 'left', 'FontSize', 9);
start = tic;
iterationG = 0;
psnrWindow = []; 
ssimWindow = []; 
lpipsWindow = []; 
min_phase1_iter = 120000;     
max_phase1_iter = 60000;    
min_phase2_iter = 40000;     
max_phase2_iter = 60000;    
min_psnr = 37;              
target_psnr = 38;           
phase3_quality_threshold = 38.5; 
phase2_start_iter = -1;     
window_size = 20;           
trainingPhase = 1;          
max_psnr = -Inf;      
max_ssim = -Inf;      
min_lpips = Inf;      
max_psnr_iter = 0;    
max_ssim_iter = 0;    
min_lpips_iter = 0;   
vggNetLPIPS = createVGGForLPIPS();
while iterationG < numIterationsG
    iterationG = iterationG + 1;
    if mod(iterationG, 2000) == 0
        elapsedTime = toc(start);
        avgTimePerIter = elapsedTime / iterationG;
        remainingTime = (numIterationsG - iterationG) * avgTimePerIter;
        fprintf('progress: %d/%d (%.1f%%) | Expected remaining time: %s\n', ...
            iterationG, numIterationsG, iterationG/numIterationsG*100, ...
            datestr(seconds(remainingTime), 'HH:MM:SS'));
    end
    if ismember(1, ablationConfig.trainingPhases) && ...
            (iterationG <= min_phase1_iter || ... 
            (iterationG <= max_phase1_iter && ...
            (isempty(psnrWindow) || mean(psnrWindow) < target_psnr)))
        trainingPhase = 1;
        phase_reason = "Reconstruction phase: Configuration requirements";
    elseif ismember(2, ablationConfig.trainingPhases) && ...
            iterationG > min_phase1_iter && ...
            (iterationG <= max_phase1_iter + max_phase2_iter)
        trainingPhase = 2;
        phase_reason = "Optical flow stage: configuration requirements";
    elseif ismember(3, ablationConfig.trainingPhases) && ...
            iterationG > max_phase1_iter + max_phase2_iter
        trainingPhase = 3;
        phase_reason = "Adversarial phase: Configuration requirements";
    else
        trainingPhase = 1;
        phase_reason = "Reconstruction phase: default";
    end
    if mod(iterationG, 5000) == 0
        save(fullfile(resultDir, 'loss_data.mat'), 'lossData');
        fprintf('The training loss data has been saved to: %s\n', fullfile(resultDir, 'loss_data.mat'));
    end
    switch trainingPhase
        case 1 
            updateDiscriminators = false;
            updateFlowNet = false;
            currentLR_G = initialLearnRateG;
            currentLR_D = 0;
            currentLR_F = 0;
        case 2 
            updateDiscriminators = mod(iterationG, 4) == 0;
            updateFlowNet = mod(iterationG, 2) == 0;
            currentLR_G = initialLearnRateG * (0.95^(iterationG/1000));
            currentLR_D = initialLearnRateD * (0.95^(iterationG/1000));
            currentLR_F = initialLearnRateF;  
        case 3 
            D_update_freq = 3;
            if extractdata(lossD) < 0.5
                D_update_freq = max(5, D_update_freq + 1);
            elseif extractdata(lossD) > 1.5
                D_update_freq = max(1, D_update_freq - 1);
            end
            updateDiscriminators = mod(iterationG, D_update_freq) == 0;
            updateFlowNet = true;
            if iterationG < 10000
                currentLR_G = initialLearnRateG * (iterationG/10000);
                currentLR_D = initialLearnRateD * (iterationG/10000);
                currentLR_F = initialLearnRateF * (iterationG/10000);
            elseif iterationG < 40000
                decay = (1 - (iterationG-10000)/(numIterationsG-10000))^0.9;
                currentLR_G = initialLearnRateG * decay;
                currentLR_D = initialLearnRateD * decay;
                currentLR_F = initialLearnRateF * decay;
            else
                decay = 0.5 * (1 + cos(pi * (iterationG-40000)/(numIterationsG-40000)));
                currentLR_G = initialLearnRateG * 0.1 * decay;
                currentLR_D = initialLearnRateD * 0.1 * decay;
                currentLR_F = initialLearnRateF * 0.1 * decay;
            end
    end
    if ~hasdata(mbqBlur) || ~hasdata(mbqSharp)
        reset(mbqBlur);
        reset(mbqSharp);
    end
    XBlur = next(mbqBlur);
    XSharp = next(mbqSharp); 
    %% Stage Generator
    switch trainingPhase
        case 1 
            [lossG, gradientsG] = dlfeval(@modelLossG_phase1, netG, vggNet, XBlur, XSharp, featureLayer); 
        case 2
            [lossG, gradientsG] = dlfeval(@modelLossG_phase2, netG, flowNet, vggNet, XBlur, XSharp, featureLayer);  
        case 3 
            [lossG, gradientsG] = dlfeval(@modelLossG, netG, netD1, netD2, netD3, flowNet, vggNet, XBlur, XSharp, iterationG, ablationConfig);
    end
    if any(cellfun(@(g) any(isnan(g(:))) || any(isinf(g(:))), gradientsG.Value))
        warning('The gradient contains NaN or Inf values, skip this update');
        continue;
    end
    gradientsG = dlupdate(@(g) clipGradient(g, 0.1), gradientsG);
    [netG, trailingAvgG, trailingAvgSqG] = adamupdate(...
        netG, gradientsG, trailingAvgG, trailingAvgSqG, iterationG, ...
        currentLR_G, gradientDecayFactor, squaredGradientDecayFactor);
    %% Update Discriminator
   if updateDiscriminators
        XGenerated = forward(netG, XBlur);
        realInput1 = cat(3, XBlur, XSharp);
        fakeInput1 = cat(3, XBlur, XGenerated);
        [lossD1, ~, gradientsD1] = dlfeval(@modelLossD, netD1, realInput1, fakeInput1);
        [netD1, trailingAvgD1, trailingAvgSqD1] = adamupdate(...
            netD1, gradientsD1, trailingAvgD1, trailingAvgSqD1, iterationG, ...
            currentLR_D, gradientDecayFactor, squaredGradientDecayFactor);
        XBlur2 = avgpool(XBlur, 2, 'Stride', 2);
        XSharp2 = avgpool(XSharp, 2, 'Stride', 2);
        XGen2 = avgpool(XGenerated, 2, 'Stride', 2);
        realInput2 = cat(3, XBlur2, XSharp2);
        fakeInput2 = cat(3, XBlur2, XGen2);
        [lossD2, ~, gradientsD2] = dlfeval(@modelLossD, netD2, realInput2, fakeInput2);
        [netD2, trailingAvgD2, trailingAvgSqD2] = adamupdate(...
            netD2, gradientsD2, trailingAvgD2, trailingAvgSqD2, iterationG, ...
            currentLR_D, gradientDecayFactor, squaredGradientDecayFactor);
        XBlur3 = avgpool(XBlur, 4, 'Stride', 4);
        XSharp3 = avgpool(XSharp, 4, 'Stride', 4);
        XGen3 = avgpool(XGenerated, 4, 'Stride', 4);
        realInput3 = cat(3, XBlur3, XSharp3);
        fakeInput3 = cat(3, XBlur3, XGen3);
        [lossD3, ~, gradientsD3] = dlfeval(@modelLossD, netD3, realInput3, fakeInput3);
        [netD3, trailingAvgD3, trailingAvgSqD3] = adamupdate(...
            netD3, gradientsD3, trailingAvgD3, trailingAvgSqD3, iterationG, ...
            currentLR_D, gradientDecayFactor, squaredGradientDecayFactor);
        lossD = (lossD1 + lossD2 + lossD3) / 3;
   end
    %% Update optical flow network
    if updateFlowNet
        flowInput = cat(3, XBlur, XSharp);
        [lossF, gradientsF] = dlfeval(@modelLossF, flowNet, flowInput, XBlur, XSharp);
        gradientsF = dlupdate(@(g) clipGradient(g, 0.05), gradientsF);
        [flowNet, trailingAvgF, trailingAvgSqF] = adamupdate(...
            flowNet, gradientsF, trailingAvgF, trailingAvgSqF, iterationG, ...
            currentLR_F, gradientDecayFactor, squaredGradientDecayFactor);
    end 
    %% Monitoring and Recording
    XGenerated = forward(netG, XBlur);
    psnr_val = calculatePSNR(XGenerated, XSharp);
    ssim_val = evaluateSSIM(XGenerated, XSharp);
    lpips_val = calculateLPIPS(XGenerated, XSharp, vggNetLPIPS);
    if psnr_val > max_psnr
        max_psnr = psnr_val;
        max_psnr_iter = iterationG;
    end
    if ssim_val > max_ssim
        max_ssim = ssim_val;
        max_ssim_iter = iterationG;
    end
    if lpips_val < min_lpips
        min_lpips = lpips_val;
        min_lpips_iter = iterationG;
    end
    psnrWindow = [psnrWindow, psnr_val];
    ssimWindow = [ssimWindow, ssim_val];
    lpipsWindow = [lpipsWindow, lpips_val];
    if numel(psnrWindow) > 1000
        psnrWindow = psnrWindow(end-999:end);
        ssimWindow = ssimWindow(end-999:end);
        lpipsWindow = lpipsWindow(end-999:end);
    end
    metrics.train.iteration(end+1) = iterationG;
    metrics.train.lossG(end+1) = double(lossG);
    metrics.train.lossD(end+1) = double(lossD);
    metrics.train.lossF(end+1) = double(lossF);
    metrics.train.psnr(end+1) = psnr_val;
    metrics.train.ssim(end+1) = ssim_val;
    metrics.train.lpips(end+1) = lpips_val;
    metrics.train.phase(end+1) = trainingPhase;
    newRow = {iterationG, trainingPhase, double(lossG), double(lossD), double(lossF), psnr_val, ssim_val, lpips_val};
    metricTable = [metricTable; newRow];
    if mod(iterationG, 5000) == 0
        save(fullfile(resultDir, 'training_metrics.mat'), 'metrics');
        writetable(metricTable, fullfile(resultDir, 'training_metrics.csv'));
    end
    addpoints(lineLossG, iterationG, double(lossG));
    addpoints(lineLossD, iterationG, double(lossD));
    addpoints(lineLossF, iterationG, double(lossF));
    addpoints(linePSNR, iterationG, psnr_val);
    addpoints(lineSSIM, iterationG, ssim_val);
    addpoints(lineLPIPS, iterationG, lpips_val);
    if mod(iterationG, 100) == 0
        phaseNames = ["Reconstruction phase", "Optical flow stage", "Adversarial phase"];
        statusText = sprintf('iteration: %d/%d |phase: %s\n', iterationG, numIterationsG, phaseNames(trainingPhase));
        statusText = [statusText, sprintf('learning rate: G=%.2e, D=%.2e, F=%.2e\n', currentLR_G, currentLR_D, currentLR_F)];
        statusText = [statusText, sprintf('loss: G=%.4f, D=%.4f, F=%.4f\n', double(lossG), double(lossD), double(lossF))];
        statusText = [statusText, sprintf('PSNR: %.2f dB | SSIM: %.4f | LPIPS: %.4f\n', psnr_val, ssim_val, lpips_val)];
        set(infoText, 'String', statusText);
    end
    drawnow;
    %% Simplify the display of statistical information
    if mod(iterationG, 2000) == 0
        if numel(psnrWindow) >= 1000
            psnr_mean = mean(psnrWindow);
            psnr_min = min(psnrWindow);
            psnr_max = max(psnrWindow);
            ssim_mean = mean(ssimWindow);
            ssim_min = min(ssimWindow);
            ssim_max = max(ssimWindow);
            lpips_mean = mean(lpipsWindow);
            lpips_min = min(lpipsWindow);
            lpips_max = max(lpipsWindow);
            fprintf('Recent 1000 iterations statistics:\n');
            fprintf('  PSNR - average: %.2f dB, scope: [%.2f, %.2f] dB\n', psnr_mean, psnr_min, psnr_max);
            fprintf('  SSIM - average: %.4f, scope: [%.4f, %.4f]\n', ssim_mean, ssim_min, ssim_max);
            fprintf('  LPIPS - average: %.4f, scope: [%.4f, %.4f]\n\n', lpips_mean, lpips_min, lpips_max);
        end
    end
    %% Validate and Save
    if mod(iterationG, 2000) == 0
        numBatches = ceil(numValidationImages / 2);
        allGenerated = [];
        for batchIdx = 1:numBatches
            batchRange = (batchIdx-1)*2 + 1 : min(batchIdx*2, numValidationImages);
            batchData = validationData(:,:,:,batchRange);
            XGeneratedBatch = forward(netG, batchData);
            allGenerated = cat(4, allGenerated, extractdata(XGeneratedBatch));
        end
        I = imtile(allGenerated);
        I = rescale(I);
        image(imageAxes, I);
        axis image off;
        title(imageAxes, sprintf("Generate Image - Iteration: %d", iterationG));
        drawnow;
        save(sprintf('net_checkpoint_iter_%d.mat', iterationG), ...
            'netG', 'netD1', 'netD2', 'netD3', 'flowNet', 'iterationG');
    end
    %% Memory Cleanup
    if mod(iterationG, 2000) == 0
        clear XGenerated XBlur2 XSharp2 XGen2 realInput2 fakeInput2
        clear XBlur3 XSharp3 XGen3 realInput3 fakeInput3
        java.lang.System.gc();
    end
end
fprintf('\n===== Summary of Training Results =====\n');
fprintf('Max PSNR: %.2f dB (Appeared on iteration %d )\n', max_psnr, max_psnr_iter);
fprintf('Max SSIM: %.4f (Appeared on iteration %d )\n', max_ssim, max_ssim_iter);
fprintf('Max LPIPS: %.4f (Appeared on iteration %d )\n', min_lpips, min_lpips_iter);
fprintf('Max PSNR: %.2f dB\n', metrics.train.psnr(end));
fprintf('Max SSIM: %.4f\n', metrics.train.ssim(end));
fprintf('Max LPIPS: %.4f\n', metrics.train.lpips(end));
if ~exist('resultDir', 'var') || isempty(resultDir)
    resultDir = fullfile(pwd, 'paper_results');
    if ~exist(resultDir, 'dir')
        mkdir(resultDir);
    end
end
if exist('metrics', 'var')
    generateOverallSummary(metrics, resultDir);
else
    warning('metrics Variable does not exist, unable to generate summary report');
end
if exist('netG', 'var') && exist('imdsBlurTest', 'var') && exist('imdsSharpTest', 'var')
    evaluateFinalModel(netG, imdsBlurTest, imdsSharpTest, resultDir, vggNetLPIPS);
else
    warning('Lack of necessary network or data variables for final evaluation');
end
if exist('netG', 'var')
    save(fullfile(resultDir, 'final_model.mat'), ...
        'netG', 'netD1', 'netD2', 'netD3', 'flowNet', 'metrics', '-v7.3');
    fprintf('The final model has been saved to: %s\n', fullfile(resultDir, 'final_model.mat'));
end
%% LPIPS
function lpips_val = calculateLPIPS(gen, ref, vggNet)
    gen_01 = (extractdata(gen) + 1) / 2;
    ref_01 = (extractdata(ref) + 1) / 2;
    gen_01 = max(0, min(1, gen_01));
    ref_01 = max(0, min(1, ref_01));
    gen_resized = imresize(gen_01, [224, 224]);
    ref_resized = imresize(ref_01, [224, 224]);
    gen_resized = single(gen_resized);
    ref_resized = single(ref_resized);
    meanImg = [123.68, 116.78, 103.94];
    for c = 1:3
        gen_resized(:,:,c,:) = gen_resized(:,:,c,:) * 255 - meanImg(c);
        ref_resized(:,:,c,:) = ref_resized(:,:,c,:) * 255 - meanImg(c);
    end
    gen_dl = dlarray(gen_resized, 'SSCB');
    ref_dl = dlarray(ref_resized, 'SSCB');
    layers = {'relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2'};
    weights = [0.1, 0.1, 0.1, 0.1, 0.1];  
    total_lpips = 0;
    for i = 1:length(layers)
        feat_gen = forward(vggNet, gen_dl, 'Outputs', layers{i});
        feat_ref = forward(vggNet, ref_dl, 'Outputs', layers{i});
        feat_gen = feat_gen ./ (sqrt(mean(feat_gen.^2, [1,2,3])) + 1e-10);
        feat_ref = feat_ref ./ (sqrt(mean(feat_ref.^2, [1,2,3])) + 1e-10);
        diff = (feat_gen - feat_ref).^2;
        layer_lpips = mean(diff, 'all');
        total_lpips = total_lpips + weights(i) * layer_lpips;
    end
    lpips_val = total_lpips;
end
%% Final evaluation function
function evaluateFinalModel(netG, testBlurDS, testSharpDS, resultDir, vggNet)
    testDir = fullfile(resultDir, 'final_test');
    if ~exist(testDir, 'dir')
        mkdir(testDir);
    end
    numTest = min([numel(testBlurDS.Files), numel(testSharpDS.Files), 50]);
    test_psnrs = zeros(numTest, 1);
    test_ssims = zeros(numTest, 1);
    test_lpipss = zeros(numTest, 1);
    test_times = zeros(numTest, 1);
    test_images = cell(numTest, 1);
    fprintf('Start final testing and evaluation (%d )...\n', numTest);
    for i = 1:numTest
        fprintf('Processing samples %d/%d\n', i, numTest);
        try
            blurImg = readimage(testBlurDS, i);
            sharpImg = readimage(testSharpDS, i);
            blurImg = preprocessSingleImage(blurImg);
            sharpImg = preprocessSingleImage(sharpImg);
            tic;
            genImg = forward(netG, blurImg);
            genTime = toc;
            psnr_val = calculatePSNR(genImg, sharpImg);
            ssim_val = evaluateSSIM(genImg, sharpImg);
            lpips_val = calculateLPIPS(genImg, sharpImg, vggNet);
            test_psnrs(i) = psnr_val;
            test_ssims(i) = ssim_val;
            test_lpipss(i) = lpips_val;
            test_times(i) = genTime;
            fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 400]);
            subplot(1,3,1);
            imshow(imtile(extractdata(blurImg)));
            title('Input (Blurry)');
            subplot(1,3,2);
            imshow(imtile(extractdata(genImg)));
            title(sprintf('DeBlurred\nPSNR: %.2f dB | SSIM: %.4f | LPIPS: %.4f', ...
                psnr_val, ssim_val, lpips_val));
            subplot(1,3,3);
            imshow(imtile(extractdata(sharpImg)));
            title('Ground Truth');
            saveas(fig, fullfile(testDir, sprintf('test_result_%02d.png', i)));
            close(fig);
            test_images{i} = struct(...
                'blur', extractdata(blurImg), ...
                'gen', extractdata(genImg), ...
                'sharp', extractdata(sharpImg), ...
                'psnr', psnr_val, ...
                'ssim', ssim_val, ...
                'lpips', lpips_val, ...
                'time', genTime);
        catch ME
            warning('Processing samples %d error: %s', i, ME.message);
            test_psnrs(i) = NaN;
            test_ssims(i) = NaN;
            test_lpipss(i) = NaN;
            test_times(i) = NaN;
        end
    end
    valid_idx = ~isnan(test_psnrs);
    test_psnrs = test_psnrs(valid_idx);
    test_ssims = test_ssims(valid_idx);
    test_lpipss = test_lpipss(valid_idx);
    test_times = test_times(valid_idx);
    test_images = test_images(valid_idx);
    avg_psnr = mean(test_psnrs);
    avg_ssim = mean(test_ssims);
    avg_lpips = mean(test_lpipss);
    avg_time = mean(test_times);
    std_psnr = std(test_psnrs);
    std_ssim = std(test_ssims);
    std_lpips = std(test_lpipss);
    save(fullfile(testDir, 'final_metrics.mat'), ...
        'test_psnrs', 'test_ssims', 'test_lpipss', 'test_times', 'test_images', ...
        'avg_psnr', 'std_psnr', 'avg_ssim', 'std_ssim', 'avg_lpips', 'std_lpips', 'avg_time');
    generateFinalReport(avg_psnr, std_psnr, avg_ssim, std_ssim, avg_lpips, std_lpips, ...
        avg_time, test_psnrs, test_ssims, test_lpipss, testDir);
    fprintf('Final testing completed。PSNR: %.2f ± %.2f dB, SSIM: %.4f ± %.4f, LPIPS: %.4f ± %.4f\n', ...
        avg_psnr, std_psnr, avg_ssim, std_ssim, avg_lpips, std_lpips);
end
%% Final report generation function
function generateFinalReport(avg_psnr, std_psnr, avg_ssim, std_ssim, avg_lpips, std_lpips, ...
    avg_time, test_psnrs, test_ssims, test_lpipss, saveDir)
    report = sprintf(['Final Evaluation Report\n' ...
                     '======================\n' ...
                     'Evaluation Date: %s\n\n' ...
                     'Performance Metrics:\n' ...
                     '  PSNR: %.2f ± %.2f dB (higher is better)\n' ...
                     '  SSIM: %.4f ± %.4f (higher is better)\n' ...
                     '  LPIPS: %.4f ± %.4f (lower is better)\n' ...
                     '  Inference Time: %.3f seconds/image\n\n' ...
                     'Statistical Significance:\n' ...
                     '  PSNR range: [%.2f, %.2f] dB\n' ...
                     '  SSIM range: [%.4f, %.4f]\n' ...
                     '  LPIPS range: [%.4f, %.4f]\n'], ...
        datetime('now'), ...
        avg_psnr, std_psnr, ...
        avg_ssim, std_ssim, ...
        avg_lpips, std_lpips, ...
        avg_time, ...
        avg_psnr - std_psnr, avg_psnr + std_psnr, ...
        avg_ssim - std_ssim, avg_ssim + std_ssim, ...
        avg_lpips - std_lpips, avg_lpips + std_lpips);
    fid = fopen(fullfile(saveDir, 'performance_report.txt'), 'w');
    fprintf(fid, '%s', report);
    fclose(fid);
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 400]);
    subplot(1,3,1);
    histogram(test_psnrs, 10, 'FaceColor', [0.2, 0.6, 0.9]);
    title(sprintf('PSNR Distribution (\\mu=%.2f, \\sigma=%.2f)', avg_psnr, std_psnr));
    xlabel('PSNR (dB)');
    ylabel('Count');
    grid on;
    subplot(1,3,2);
    histogram(test_ssims, 10, 'FaceColor', [0.9, 0.6, 0.2]);
    title(sprintf('SSIM Distribution (\\mu=%.4f, \\sigma=%.4f)', avg_ssim, std_ssim));
    xlabel('SSIM');
    ylabel('Count');
    grid on;
    subplot(1,3,3);
    histogram(test_lpipss, 10, 'FaceColor', [0.6, 0.2, 0.8]);
    title(sprintf('LPIPS Distribution (\\mu=%.4f, \\sigma=%.4f)', avg_lpips, std_lpips));
    xlabel('LPIPS');
    ylabel('Count');
    grid on;
    saveas(fig, fullfile(saveDir, 'metrics_distribution.png'));
    close(fig);
    fig2 = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
    norm_psnr = (test_psnrs - min(test_psnrs)) / (max(test_psnrs) - min(test_psnrs));
    norm_ssim = (test_ssims - min(test_ssims)) / (max(test_ssims) - min(test_ssims));
    norm_lpips = 1 - ((test_lpipss - min(test_lpipss)) / (max(test_lpipss) - min(test_lpipss))); % 反转LPIPS
    plot(1:length(test_psnrs), norm_psnr, 'o-', 'LineWidth', 2, 'Color', [0.2, 0.6, 0.9]);
    hold on;
    plot(1:length(test_ssims), norm_ssim, 's-', 'LineWidth', 2, 'Color', [0.9, 0.6, 0.2]);
    plot(1:length(test_lpipss), norm_lpips, 'd-', 'LineWidth', 2, 'Color', [0.6, 0.2, 0.8]);
    hold off;
    title('Comparison of Standardization Indicators');
    xlabel('Image sequence number');
    ylabel('Standardized value (higher, better)');
    legend('PSNR', 'SSIM', '1-LPIPS');
    grid on;
    saveas(fig2, fullfile(saveDir, 'metrics_comparison.png'));
    close(fig2);
    fprintf('\n=====Final test results =====\n%s\n', report);
end
%% Comprehensive summary function
function generateOverallSummary(metrics, saveDir)
    summaryDir = fullfile(saveDir, 'summary_charts');
    if ~exist(summaryDir, 'dir')
        mkdir(summaryDir);
    end
    max_iter = max(metrics.train.iteration);
    final_psnr = metrics.train.psnr(end);
    final_ssim = metrics.train.ssim(end);
    final_lpips = metrics.train.lpips(end);
    if isfield(metrics, 'val') && ~isempty(metrics.val.psnr)
        best_val_psnr = max(metrics.val.psnr);
        best_val_ssim = max(metrics.val.ssim);
        best_val_lpips = min(metrics.val.lpips);
        best_val_iter = metrics.val.iteration(metrics.val.psnr == best_val_psnr);
        if numel(best_val_iter) > 1
            best_val_iter = best_val_iter(1);
        end
    else
        best_val_psnr = NaN;
        best_val_ssim = NaN;
        best_val_lpips = NaN;
        best_val_iter = NaN;
    end
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
    subplot(2,2,1);
    plot(metrics.train.iteration, metrics.train.lossG, 'LineWidth', 1.5);
    hold on;
    if isfield(metrics.train, 'lossD') && ~isempty(metrics.train.lossD)
        plot(metrics.train.iteration, metrics.train.lossD, 'LineWidth', 1.5);
    end
    if isfield(metrics.train, 'lossF') && ~isempty(metrics.train.lossF)
        plot(metrics.train.iteration, metrics.train.lossF, 'LineWidth', 1.5);
    end
    hold off;
    title('Training Losses');
    xlabel('Iteration');
    ylabel('Loss');
    legend('Generator', 'Discriminator', 'Flow');
    grid on;
    subplot(2,2,2);
    plot(metrics.train.iteration, metrics.train.psnr, 'LineWidth', 2, 'Color', [0, 0.5, 0]);
    hold on;
    plot(metrics.train.iteration, metrics.train.ssim, 'LineWidth', 2, 'Color', [0.85, 0.33, 0.1]);
    plot(metrics.train.iteration, metrics.train.lpips, 'LineWidth', 2, 'Color', [0.49, 0.18, 0.56]);
    hold off;
    title('Training Metrics');
    xlabel('Iteration');
    ylabel('Metric Value');
    legend('PSNR', 'SSIM', 'LPIPS');
    grid on;
    subplot(2,2,3);
    if isfield(metrics.train, 'phase') && ~isempty(metrics.train.phase)
        phases = metrics.train.phase;
        plot(metrics.train.iteration, phases, 'LineWidth', 2);
        title('Training Phase Transition');
        xlabel('Iteration');
        ylabel('Phase');
        yticks([1,2,3]);
        yticklabels({'Reconstruction', 'Optical Flow', 'Adversarial'});
        grid on;
    else
        text(0.5, 0.5, 'Phase data not available', 'HorizontalAlignment', 'center');
        title('Training Phase Transition (Data Not Available)');
    end
    subplot(2,2,4);
    if isfield(metrics, 'val') && ~isempty(metrics.val.psnr)
        plot(metrics.val.iteration, metrics.val.psnr, 'o-', 'LineWidth', 2, 'Color', [0, 0.5, 0]);
        hold on;
        plot(metrics.val.iteration, metrics.val.ssim, 's-', 'LineWidth', 2, 'Color', [0.85, 0.33, 0.1]);
        plot(metrics.val.iteration, metrics.val.lpips, 'd-', 'LineWidth', 2, 'Color', [0.49, 0.18, 0.56]);
        hold off;
        title('Validation Metrics');
        xlabel('Iteration');
        ylabel('Metric Value');
        legend('PSNR', 'SSIM', 'LPIPS');
        grid on;
    else
        text(0.5, 0.5, 'Validation data not available', 'HorizontalAlignment', 'center');
        title('Validation Metrics (Data Not Available)');
    end
    saveas(fig, fullfile(summaryDir, 'training_summary.png'));
    close(fig);
    if isfield(metrics, 'val') && ~isempty(metrics.val.psnr)
        fig2 = figure('Visible', 'off', 'Position', [100, 100, 1200, 400]);
        subplot(1,3,1);
        histogram(metrics.val.psnr, 10, 'FaceColor', [0.2, 0.6, 0.9]);
        title(sprintf('Validation PSNR Distribution\n(\\mu=%.2f, \\sigma=%.2f)', ...
            mean(metrics.val.psnr), std(metrics.val.psnr)));
        xlabel('PSNR (dB)');
        ylabel('Count');
        grid on;
        subplot(1,3,2);
        histogram(metrics.val.ssim, 10, 'FaceColor', [0.9, 0.6, 0.2]);
        title(sprintf('Validation SSIM Distribution\n(\\mu=%.4f, \\sigma=%.4f)', ...
            mean(metrics.val.ssim), std(metrics.val.ssim)));
        xlabel('SSIM');
        ylabel('Count');
        grid on;
        subplot(1,3,3);
        histogram(metrics.val.lpips, 10, 'FaceColor', [0.6, 0.2, 0.8]);
        title(sprintf('Validation LPIPS Distribution\n(\\mu=%.4f, \\sigma=%.4f)', ...
            mean(metrics.val.lpips), std(metrics.val.lpips)));
        xlabel('LPIPS');
        ylabel('Count');
        grid on;
        saveas(fig2, fullfile(summaryDir, 'validation_distribution.png'));
        close(fig2);
    end
    [max_psnr, max_psnr_idx] = max(metrics.train.psnr);
    [max_ssim, max_ssim_idx] = max(metrics.train.ssim);
    [min_lpips, min_lpips_idx] = min(metrics.train.lpips);
    summary = sprintf(['===== Thesis Abstract Report =====\n' ...
                      'generation time: %s\n' ...
                      'Total number of iterations: %d\n' ...
                      'Final training PSNR: %.2f dB\n' ...
                      'Final training SSIM: %.4f\n' ...
                      'Final training of LPIPS: %.4f\n'], ...
        datetime('now'), max_iter, final_psnr, final_ssim, final_lpips);
    if ~isnan(best_val_psnr)
        summary = [summary, sprintf('Best Verification PSNR: %.2f dB (迭代 %d)\n', best_val_psnr, best_val_iter)];
        summary = [summary, sprintf('Best Verification SSIM: %.4f\n', best_val_ssim)];
        summary = [summary, sprintf('Best Verification LPIPS: %.4f\n', best_val_lpips)];
    end
    summary = [summary, sprintf('The maximum during the training process PSNR: %.2f dB (迭代 %d)\n', max_psnr, metrics.train.iteration(max_psnr_idx))];
    summary = [summary, sprintf('The maximum during the training processSSIM: %.4f (迭代 %d)\n', max_ssim, metrics.train.iteration(max_ssim_idx))];
    summary = [summary, sprintf('The minimum during the training process LPIPS: %.4f (迭代 %d)\n', min_lpips, metrics.train.iteration(min_lpips_idx))];
    summary = [summary, sprintf('\nModel Architecture:\n')];
    summary = [summary, sprintf('  - Enhanced U-Net Generator (8 Residual Attention Blocks)\n')];
    summary = [summary, sprintf('  - Three level multi-scale discriminator\n')];
    summary = [summary, sprintf('  - Adaptive Optical Flow Network\n')];
    summary = [summary, sprintf('\ntraining strategy:\n')];
    summary = [summary, sprintf('  - Three-stage progressive training\n')];
    summary = [summary, sprintf('  - Adaptive learning rate scheduling\n')];
    summary = [summary, sprintf('  - Mixed loss function (L1+perception+adversarial+optical flow)\n')];
    fid = fopen(fullfile(saveDir, 'paper_summary.txt'), 'w');
    fprintf(fid, '%s', summary);
    fclose(fid);
    disp(summary);
    disp(['The detailed report has been saved to: ' fullfile(saveDir, 'paper_summary.txt')]);
    disp(['The chart has been saved to: ' summaryDir]);
end
%% ==================== Helper function ====================
function dlX = preprocessSingleImage(data)
    if iscell(data)
        numImages = numel(data);
        processed = cell(1, numImages);
        for i = 1:numImages
            img = data{i};
            if size(img, 3) == 1
                img = repmat(img, 1, 1, 3);
            end
            img = imresize(img, [256, 256]);
            if isa(img, 'uint8')
                img = single(img)/255.0;
            end
            img = (img * 2) - 1;
            processed{i} = img;
        end
        X = cat(4, processed{:});
    else
        X = data;
        if size(X, 3) == 1
            X = repmat(X, 1, 1, 3);
        end
        X = imresize(X, [256, 256]);
        if isa(X, 'uint8')
            X = single(X)/255.0;
        end
        X = (X * 2) - 1;
    end

    X(isnan(X)) = 0;
    X(isinf(X)) = 0;
    dlX = dlarray(X, 'SSCB');
end
function g = clipGradient(g, threshold)
    if ~isempty(g)
        norm_g = sqrt(sum(g.^2, 'all'));
        adaptive_threshold = min(1.0, max(0.05, threshold * (1 + 0.05*log(norm_g+1))));
        if norm_g > adaptive_threshold
            g = g * (adaptive_threshold / (norm_g + 1e-8));
        end
        noise_scale = 1e-3;
        g = g + noise_scale * randn(size(g), 'like', g);
    end
end
function percep_loss = safePercepLoss(vggNet, gen, ref, featureLayer)
    gen_01 = (extractdata(gen) + 1) / 2;
    ref_01 = (extractdata(ref) + 1) / 2;
    gen_01 = imresize(gen_01, [224, 224]);
    ref_01 = imresize(ref_01, [224, 224]);
    gen_dl = dlarray(gen_01, 'SSCB');
    ref_dl = dlarray(ref_01, 'SSCB');
    feat_gen = forward(vggNet, gen_dl); 
    feat_ref = forward(vggNet, ref_dl);
    percep_loss = mean(abs(feat_gen - feat_ref), 'all');
end
function mssim = multissim(img1, img2)
    img1 = max(0, min(1, img1));
    img2 = max(0, min(1, img2));
    dims = ndims(img1);
    if dims == 4 
        numImages = size(img1,4);
        ssim_vals = zeros(1, numImages);
        for i = 1:numImages
            img1_single = img1(:,:,:,i);
            img2_single = img2(:,:,:,i);
            if size(img1_single, 3) == 1
                img1_single = squeeze(img1_single);
                img2_single = squeeze(img2_single);
            end
            ssim_val = ssim(img1_single, img2_single);
            if isscalar(ssim_val)
                ssim_vals(i) = ssim_val;
            else
                ssim_vals(i) = mean(ssim_val(:));
            end
        end
        mssim = mean(ssim_vals);
    elseif dims == 3 
        ssim_val = ssim(img1, img2);
        if isscalar(ssim_val)
            mssim = ssim_val;
        else
            mssim = mean(ssim_val(:));
        end 
    elseif dims == 2 
        mssim = ssim(img1, img2);
    else
        error('Unsupported image dimensions: %d', dims);
    end
end
function vggNet = createLightweightFeatureExtractor()
    layers = [
        imageInputLayer([256 256 3], 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(3, 32, 'Padding','same','Name','conv1')
        reluLayer('Name','relu1')
        maxPooling2dLayer(2,'Stride',2,'Name','pool1')
        convolution2dLayer(3, 64, 'Padding','same','Name','conv2')
        reluLayer('Name','relu2')
        maxPooling2dLayer(2,'Stride',2,'Name','pool2')
        convolution2dLayer(3, 128, 'Padding','same','Name','conv3')
        reluLayer('Name','relu3')
        ];
    lgraph = layerGraph(layers);
    vggNet = dlnetwork(lgraph);
    featureLayer = 'relu3'; 
end
function [lossG, gradientsG] = modelLossG_phase2(netG, flowNet, vggNet, XBlur, XSharp, featureLayer)
    XGenerated = forward(netG, XBlur);
    l1_loss = 5.0 * mean(abs(XGenerated - XSharp), 'all');
    flowInput = cat(3, XBlur, XGenerated);
    flowField = forward(flowNet, flowInput);
    warpedSharp = warpImage(XSharp, flowField);
    flow_loss = 0.7 * mean(abs(warpedSharp - XBlur), 'all') + ...
                0.3 * mean(abs(XGenerated - warpedSharp), 'all');
    percep_loss = safePercepLoss(vggNet, XGenerated, XSharp, featureLayer);
    lossG = l1_loss + flow_loss + 0.5 * percep_loss;
    gradientsG = dlgradient(lossG, netG.Learnables);
end
function warped = warpImage(img, flow)
    img_data = extractdata(img);
    flow_data = extractdata(flow);
    [H, W, C, N] = size(img_data);
    [X, Y] = meshgrid(1:W, 1:H);
    warped = zeros(size(img_data), 'like', img_data);
    for n = 1:N
        for c = 1:C
            current_img = img_data(:,:,c,n);
            U = flow_data(:,:,1,n); 
            V = flow_data(:,:,2,n); 
            Xw = X + U;
            Yw = Y + V;
            Xw = min(max(Xw, 1.5), W-0.5);
            Yw = min(max(Yw, 1.5), H-0.5);
            warped(:,:,c,n) = interp2(X, Y, current_img, Xw, Yw, 'linear', 0);
        end
    end
    warped = dlarray(warped, 'SSCB');
end
function [lossF, gradientsF] = modelLossF(flowNet, flowInput, XBlur, XSharp)
flowField = forward(flowNet, flowInput);
warpedSharp = warpImage(XSharp, flowField);
photometricLoss = mean(abs(warpedSharp - XBlur).^0.85, 'all');
flow_backward = forward(flowNet, cat(3, XSharp, XBlur));
cycle_loss = mean(abs(flowField + warpImage(flow_backward, flowField)), 'all');
[dx, dy] = gradient(extractdata(XBlur));
edge_weight = exp(-sqrt(dx.^2 + dy.^2));
edge_weight = dlarray(edge_weight, 'SSCB');
flow_x = flowField(:,:,1,:);
flow_y = flowField(:,:,2,:);
[dx_flow, dy_flow] = gradient(flow_x);
smoothLoss = 0.1 * mean(edge_weight .* (dx_flow.^2 + dy_flow.^2), 'all');
lossF = photometricLoss + smoothLoss + 0.3*cycle_loss;
gradientsF = dlgradient(lossF, flowNet.Learnables);
gradientsF = dlupdate(@(g) clipGradient(g, 0.1), gradientsF);
end
function [lossD, lossDUnregularized, gradientsD] = modelLossD(netD, realInput, fakeInput)
    YReal = forward(netD, realInput);
    YFake = forward(netD, fakeInput);
    loss_real = mean((YReal - 1).^2, 'all'); 
    loss_fake = mean((YFake - 0).^2, 'all'); 
    lossDUnregularized = 0.5 * (loss_real + loss_fake);
    num_critic = 1; 
    penalty = 0;
    lambda = 1.0;  
    epsilon = rand(1,1,1,size(realInput,4),'like',realInput);
    mixedInput = epsilon.*realInput + (1-epsilon).*fakeInput;
    mixedOutput = forward(netD, mixedInput);
    mixedOutputMean = mean(mixedOutput,'all');
    gradients = dlgradient(mixedOutputMean, mixedInput);
    gradients_norm = sqrt(sum(gradients.^2, [1,2,3]) + 1e-8);
    penalty = lambda * mean((gradients_norm - 1).^2, 'all');
    lossD = lossDUnregularized + penalty;
    gradientsD = dlgradient(lossD, netD.Learnables);
end
%% 
function [lossG, gradientsG] = modelLossG(netG, netD1, netD2, netD3, flowNet, vggNet, XBlur, XSharp, iteration, ablationConfig)
    if nargin < 10 || isempty(ablationConfig)
        ablationConfig = struct();
        ablationConfig.useFlowNet = true;
        ablationConfig.usePerceptualLoss = true;
    end
    XGenerated = forward(netG, XBlur);
    adv_loss = 0;
    fakeInput1 = cat(3, XBlur, XGenerated);
    YPred1 = forward(netD1, fakeInput1);
    adv_loss = adv_loss + mean((YPred1 - 1).^2, 'all');
    XBlur2 = avgpool(XBlur, 2, 'Stride', 2);
    XGen2 = avgpool(XGenerated, 2, 'Stride', 2);
    fakeInput2 = cat(3, XBlur2, XGen2);
    YPred2 = forward(netD2, fakeInput2);
    adv_loss = adv_loss + mean((YPred2 - 1).^2, 'all');
    XBlur3 = avgpool(XBlur, 4, 'Stride', 4);
    XGen3 = avgpool(XGenerated, 4, 'Stride', 4);
    fakeInput3 = cat(3, XBlur3, XGen3);
    YPred3 = forward(netD3, fakeInput3);
    adv_loss = adv_loss + mean((YPred3 - 1).^2, 'all');
    l1_loss = 8.0 * mean(abs(XGenerated - XSharp), 'all') + 0.01 * mean(abs(XGenerated), 'all');
    if isfield(ablationConfig, 'useFlowNet') && ablationConfig.useFlowNet
        flowInput = cat(3, XBlur, XGenerated);
        flowField = forward(flowNet, flowInput);
        warpedSharp = warpImage(XSharp, flowField);
        flow_loss = 0.5 * mean(abs(warpedSharp - XBlur), 'all') + ...
                    0.2 * mean(abs(XGenerated - warpedSharp), 'all');
    else
        flow_loss = 0;
    end
    if isfield(ablationConfig, 'usePerceptualLoss') && ablationConfig.usePerceptualLoss
        percep_loss = enhancedPercepLoss(vggNet, XGenerated, XSharp);
    else
        percep_loss = 0;
    end
    XGen01 = (extractdata(XGenerated) + 1) / 2;
    XSharp01 = (extractdata(XSharp) + 1) / 2;
    
    if canUseGPU()
        XGen01 = gather(XGen01);
        XSharp01 = gather(XSharp01);
    end    
    ssim_val = multissim(XGen01, XSharp01);
    ssim_loss = 2.0 * (1 - ssim_val);
    noise_scale = 0.02;
    XBlur_noisy = XBlur + noise_scale * randn(size(XBlur), 'like', XBlur);
    XGenerated_noisy = forward(netG, XBlur_noisy);
    consistency_loss = 0.1 * mean(abs(XGenerated - XGenerated_noisy), 'all');
    if iteration > 40000
        adv_weight = 0.05;
        l1_weight = 8.0;
        flow_weight = 0.2;
        percep_weight = 0.2;
        ssim_weight = 2.0;
    else
        adv_weight = 0.1;
        l1_weight = 5.0;
        flow_weight = 0.3;
        percep_weight = 0.2;
        ssim_weight = 1.5;
    end
    total_loss = ...
        adv_weight * adv_loss + ...
        l1_weight * l1_loss + ...
        flow_weight * flow_loss + ...
        percep_weight * percep_loss + ...
        ssim_weight * ssim_loss + ...
        consistency_loss;
    gradientsG = dlgradient(total_loss, netG.Learnables);
    lossG = total_loss;
end
%% Upgrade of multi-scale feature extraction network
function vggNet = createEnhancedFeatureExtractor()
    net = vgg19('Weights', 'imagenet');
    layers = net.Layers(1:25); 
    lgraph = layerGraph(layers);
    vggNet = dlnetwork(lgraph);
end
function lgraph = spatialAttentionLayer(lgraph, inputLayerName, layerName)
    gap = globalAveragePooling2dLayer('Name', [layerName '_gap']);
    gmp = globalMaxPooling2dLayer('Name', [layerName '_gmp']);
    concat = concatenationLayer(3, 2, 'Name', [layerName '_concat']);
    conv = convolution2dLayer(7, 1, 'Padding', 'same', 'Name', [layerName '_conv']);
    sigmoid = sigmoidLayer('Name', [layerName '_sigmoid']);
    mult = multiplicationLayer(2, 'Name', [layerName '_mult']);
    lgraph = addLayers(lgraph, gap);
    lgraph = addLayers(lgraph, gmp);
    lgraph = addLayers(lgraph, concat);
    lgraph = addLayers(lgraph, conv);
    lgraph = addLayers(lgraph, sigmoid);
    lgraph = addLayers(lgraph, mult);
    lgraph = safeConnect(lgraph, inputLayerName, [layerName '_gap/in']);
    lgraph = safeConnect(lgraph, inputLayerName, [layerName '_gmp/in']);
    lgraph = safeConnect(lgraph, [layerName '_gap'], [layerName '_concat/in1']);
    lgraph = safeConnect(lgraph, [layerName '_gmp'], [layerName '_concat/in2']);
    lgraph = safeConnect(lgraph, [layerName '_concat'], [layerName '_conv']);
    lgraph = safeConnect(lgraph, [layerName '_conv'], [layerName '_sigmoid']);
    lgraph = safeConnect(lgraph, inputLayerName, [layerName '_mult/in1']);
    lgraph = safeConnect(lgraph, [layerName '_sigmoid'], [layerName '_mult/in2']);
end
function psnr_val = calculatePSNR(gen, ref)
    gen_01 = (extractdata(gen) + 1) / 2;
    ref_01 = (extractdata(ref) + 1) / 2;
    gen_01 = max(0, min(1, gen_01));
    ref_01 = max(0, min(1, ref_01));
    mse = mean((gen_01 - ref_01).^2, 'all');
    psnr_val = 10 * log10(1 / (mse + 1e-10));
end
function percep_loss = enhancedPercepLoss(vggNet, gen, ref)
    layers = {'relu3_1', 'relu4_1', 'relu5_1'};
    weights = [0.2, 0.3, 0.5];
    percep_loss = 0;
    gen_01 = (extractdata(gen) + 1) / 2;
    ref_01 = (extractdata(ref) + 1) / 2;
    for i = 1:numel(layers)
        gen_resized = imresize(gen_01, [224, 224]);
        ref_resized = imresize(ref_01, [224, 224]);
        gen_resized = single(gen_resized);
        ref_resized = single(ref_resized);
        gen_dl = dlarray(gen_resized, 'SSCB');
        ref_dl = dlarray(ref_resized, 'SSCB');
        feat_gen = forward(vggNet, gen_dl, 'Outputs', layers{i});
        feat_ref = forward(vggNet, ref_dl, 'Outputs', layers{i});
        layer_loss = mean(abs(feat_gen - feat_ref), 'all');
        percep_loss = percep_loss + weights(i) * layer_loss;
    end
end
function vggNet = loadVGG19FromMat()
    if ~exist('vgg19.mat', 'file')
        error('vgg19.matFile not found. Please ensure that the file exists in the current directory or MATLAB path。');
    end
    vggData = load('vgg19.mat');
    if isfield(vggData, 'net')
        net = vggData.net;
    elseif isfield(vggData, 'vgg19')
        net = vggData.vgg19;
    else
        fieldNames = fieldnames(vggData);
        netFound = false;
        for i = 1:length(fieldNames)
            if isa(vggData.(fieldNames{i}), 'SeriesNetwork') || ...
               isa(vggData.(fieldNames{i}), 'DAGNetwork') || ...
               isa(vggData.(fieldNames{i}), 'dlnetwork')
                net = vggData.(fieldNames{i});
                netFound = true;
                break;
            end
        end
        if ~netFound
            error('VGG19.mat No valid network structure was found in the file。');
        end
    end
    if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
        layers = net.Layers;
    elseif isa(net, 'dlnetwork')
        layers = net.Layers;
    else
        error('Unsupported network type: %s', class(net));
    end
    targetLayerName = 'relu5_4';
    layerNames = arrayfun(@(l) l.Name, layers, 'UniformOutput', false);
    targetIdx = find(strcmp(layerNames, targetLayerName), 1, 'last');
    if isempty(targetIdx)
        convLayers = find(cellfun(@(x) contains(x, 'conv') || contains(x, 'relu'), layerNames));
        if ~isempty(convLayers)
            targetIdx = convLayers(end);
            fprintf('Warning: Not found%s，use%s to \n', targetLayerName, layerNames{targetIdx});
        else
            error('No suitable feature layer found。');
        end
    end
    subLayers = layers(1:targetIdx);
    lgraph = layerGraph(subLayers);
    vggNet = dlnetwork(lgraph);
    fprintf('VGG19Network loading successful. feature layer: %s\n', layerNames{targetIdx});
end
%% LPIPS
function vggNet = createVGGForLPIPS()
    vggNet = loadVGG19FromMat();
    fprintf('VGG19Network loading successful, used for LPIPS computation.\n');
end
function percep_loss = vggPerceptualLoss(vggNet, gen, ref)
    try
        gen_processed = preprocessForVGG(gen);
        ref_processed = preprocessForVGG(ref);
        layers = {'relu3_1', 'relu4_1', 'relu5_1'};
        percep_loss = 0;
        for i = 1:numel(layers)
            feat_gen = activations(vggNet, gen_processed, layers{i}, 'OutputAs', 'channels');
            feat_ref = activations(vggNet, ref_processed, layers{i}, 'OutputAs', 'channels');
            percep_loss = percep_loss + mean(abs(feat_gen - feat_ref), 'all');
        end
    catch ME
        fprintf('VGG perceptual loss calculation error: %s\n', ME.message);
        percep_loss = 0;
    end
end
function processed = preprocessForVGG(img)
    if ~isa(img, 'single')
        img = single(img);
    end
    img_uint8 = (img + 1) * 127.5;
    img_resized = imresize(img_uint8, [224, 224]);
    meanImg = [123.68, 116.78, 103.94];
    processed = img_resized;
    for c = 1:3
        processed(:,:,c,:) = processed(:,:,c,:) - meanImg(c);
    end
    processed = dlarray(processed, 'SSCB');
end
function isPresent = isConnectionPresent(lgraph, source, destination)
    isPresent = false;
    connections = lgraph.Connections;
    for i = 1:size(connections, 1)
        src = connections.Source{i};
        dest = connections.Destination{i};
        if strcmp(src, source) && strcmp(dest, destination)
            isPresent = true;
            return;
        end
    end
end
function lgraph = safeConnect(lgraph, source, destination)
    if ~isConnectionPresent(lgraph, source, destination)
        lgraph = connectLayers(lgraph, source, destination);
    else
        fprintf('Connection already exists: %s -> %s\n', source, destination);
    end
end

function ssim_val = evaluateSSIM(gen, ref)
    gen_01 = (extractdata(gen) + 1) / 2;
    ref_01 = (extractdata(ref) + 1) / 2;
    gen_01 = max(0, min(1, gen_01));
    ref_01 = max(0, min(1, ref_01));
    ssim_vals = zeros(size(gen_01, 4), 1);
    for i = 1:size(gen_01, 4)
        img_gen = gen_01(:, :, :, i);
        img_ref = ref_01(:, :, :, i);
        if size(img_gen, 3) == 1
            img_gen = repmat(img_gen, 1, 1, 3);
            img_ref = repmat(img_ref, 1, 1, 3);
        end
        ssim_vals(i) = ssim(img_gen, img_ref);
    end
    ssim_val = mean(ssim_vals);
end
