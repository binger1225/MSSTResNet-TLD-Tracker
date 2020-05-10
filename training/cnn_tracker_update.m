%  FILE:   cnn_tracker_update.m
%
%    This function serves as the main function for training a model.
%  
%  INPUT:  configuration     (see code for details) 
%
%  OUTPUT: net               (trained network)
%          info              (training stats)

function [net, info] = cnn_tracker_update(net, imdb, varargin)


opts.keepDilatedZeros = false;
opts.inputSize = [500, 500];
opts.learningRate = 1e-4;

%% use customized training function ie. adam
opts.trainFn = '@cnn_train_dag';
opts.batchGetterFn = '@cnn_get_batch_logistic_zoom';
opts.freezeResNet = false;
opts.tag = '';
opts.clusterNum = 25;
opts.clusterName = '';
opts.bboxReg = true;
opts.skipLRMult = [1, 0.001, 0.0001, 0.00001];
opts.sampleSize = 256;
opts.posFraction = 0.5;
opts.posThresh = 0.7;
opts.negThresh = 0.3;
opts.border = [0, 0];
opts.pretrainModelPath = 'matconvnet/pascal-fcn8s-tvg-dag.mat';
opts.dataDir = fullfile('data','widerface') ;
opts.modelType = 'pascal-fcn8s-tvg-dag' ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
opts.video = '';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.minClusterSize = [2, 2]; 
opts.maxClusterSize = opts.inputSize;
[opts, varargin] = vl_argparse(opts, varargin) ;


opts.batchSize = 10;
opts.numSubBatches = 1;
opts.numEpochs = 50;
opts.gpus = [1];
opts.numFetchThreads = 8;
opts.lite = false;
opts = vl_argparse(opts, varargin) ;

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;

opts.train.gpus = opts.gpus;
opts.train.batchSize = opts.batchSize;
opts.train.numSubBatches = opts.numSubBatches;
opts.train.numEpochs = opts.numEpochs;
opts.train.learningRate = opts.learningRate;

opts.train.keepDilatedZeros = opts.keepDilatedZeros;


trainFn = @cnn_train_dag_hardmine;
batchGetter = @cnn_get_batch_hardmine;


%% start training (no validation)
derOutputs = {'loss_cls', 1, 'loss_reg', 1}; 
[net, info] = trainFn(net, imdb, getBatchFn(batchGetter, opts, net.meta), ...
                      'derOutputs', derOutputs, ...
                      'val', nan,...
                      opts.train) ;
                  
end


%% wrapper for batch getter function
function fn = getBatchFn(batchGetter, opts, meta)
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.inputSize = meta.normalization.inputSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;

if isfield(meta, 'sampleSize'), bopts.sampleSize = meta.sampleSize; end;
if isfield(meta, 'posFraction'), bopts.posFraction = meta.posFraction; end;
if isfield(meta, 'posThresh'), bopts.posThresh = meta.posThresh; end;
if isfield(meta, 'negThresh'), bopts.negThresh = meta.negThresh; end;

if isfield(meta, 'clusters'), bopts.clusters = meta.clusters; end;
if isfield(meta, 'var2idx'), bopts.var2idx = meta.var2idx; end;
if isfield(meta, 'varsizes'), bopts.varsizes = meta.varsizes; end;
if isfield(meta, 'recfields'), bopts.recfields = meta.recfields; end;
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(batchGetter, bopts,x,y) ;
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(batchGetter, bopts,useGpu,x,y) ;
end
end

%% interface to batch getter function
function inputs = getDagNNBatch(batchGetter, opts, useGpu, imdb, batch)
imgs = imdb.images.img(batch);
labelRects = imdb.labels.rects(batch);

[images, clsmaps, regmaps] = batchGetter(imgs, labelRects, ...
                                       opts, 'prefetch', nargout == 0) ;

if nargout > 0
    % if we are training
    if useGpu && ~isempty(clsmaps) && ~isempty(regmaps)
        images = gpuArray(images) ;
    end
    inputs = {'data', images};
    
    if ~isempty(clsmaps)
        inputs(end+1:end+2) = {'label_cls', clsmaps}; 
    end
    if ~isempty(regmaps)
        inputs(end+1:end+2) = {'label_reg', regmaps};
    end
end
end



