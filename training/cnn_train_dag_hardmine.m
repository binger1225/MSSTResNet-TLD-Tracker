%  FILE:   cnn_train_dag_hardmine.m
% 
%    This function works with the get batch function and trains the detection
%    network with hard negative mining.
% 
%  INPUT:  imagePaths (image paths of a batch of images)
%          imageSizes (image sizes of the same batch of images)
%          labelRects (ground truth bounding boxes)
% 
%  OUTPUT: images (500x500 random cropped regions)
%          clsmaps (ground truth classification heat map)
%          regmaps (ground truth regression heat map)

function [net,stats] = cnn_train_dag_hardmine(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.snapshotIter = inf;

% opts.sampleSize = 512;
opts.sampleSize = 256;
opts.posFraction = 0.5;

opts.keepDilatedZeros = false;
opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = false;
opts.video = '';
opts = vl_argparse(opts, varargin) ;


if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = []; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end

state.getBatch = getBatch ;
stats = [] ;

% check if loss layer is DetLoss
if ~isa(net.layers(net.getLayerIndex('loss_cls')).block, 'dagnn.DetLoss')
    net.removeLayer('loss_cls');
    net.addLayer('loss_cls', dagnn.DetLoss('loss', 'logistic'), ...
                 {'score_cls', 'label_cls'}, 'loss_cls');
end

start =  0;
for epoch=start+1:opts.numEpochs
  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;

  % Train for one epoch.
  state.epoch = epoch ;

  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val(randperm(numel(opts.val))) ;
  state.imdb = imdb ;


  [stats.train(epoch),prof] = process_epoch(net, state, opts, 'train') ;
  stats.val(epoch) = process_epoch(net, state, opts, 'val') ;


end

% -------------------------------------------------------------------------
function [stats, prof] = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

% initialize empty momentum
if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

% move CNN  to GPU as needed
mmap = [] ;

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;

% NOTE a man wants to keep the batch size consistent
for t = 1:opts.batchSize:numel(subset) 
  fprintf('epoch %02d/%02d... \n', state.epoch, opts.numEpochs) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = state.getBatch(state.imdb, batch) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      state.getBatch(state.imdb, nextBatch) ;
    end

    if strcmp(mode, 'train') 
        net.mode = 'normal' ;
        net.accumulateParamDers = (s ~= 1) ;

        % forward pass
        net.forward(inputs, opts.derOutputs);

        % NOTE hard example selection (change cls label variable) no need to
        % change reg label because it listens to pos cls label
        loss_cls_map = net.layers(net.getLayerIndex('loss_cls')).block.loss_map;
        label_cls = net.vars(net.getVarIndex('label_cls')).value;

        % poor man's version to ensure diversity and difficulty
        label_cls(loss_cls_map<0.03) = 0;
        pos_num = 0; neg_num = 0;
        for i = 1:size(label_cls,4)
            clsmap = label_cls(:,:,:,i);
            lossmap = loss_cls_map(:,:,:,i);
            
            pos_maxnum = opts.sampleSize*opts.posFraction;
            % pos_maxnum  = 10;
            pos_idx = find(clsmap(:)==1);
            lossmap_pos = lossmap(pos_idx);
            pos_num = pos_num + numel(pos_idx);
            if numel(pos_idx) > pos_maxnum
                didx = Shuffle(numel(pos_idx), 'index', numel(pos_idx)-pos_maxnum);
                clsmap(pos_idx(didx)) = 0;
            end
            
            
            neg_maxnum =  opts.sampleSize*(1-opts.posFraction); 
            neg_idx = find(clsmap(:)==-1);
            lossmap_neg = lossmap(neg_idx);
            neg_num = neg_num + numel(neg_idx);
            if numel(neg_idx) > neg_maxnum
                ridx = Shuffle(numel(neg_idx), 'index', gather(neg_maxnum));
                didx = [1:numel(neg_idx)];
                didx(ridx) = [];
                clsmap(neg_idx(didx)) = 0;
            end
            
            label_cls(:,:,:,i) = clsmap;
        end

        net.vars(net.getVarIndex('label_cls')).value = label_cls;

        % backward pass
        net.backward(inputs, opts.derOutputs);
    else
        error('do not use this function for testing');
    end
  end

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % get statistics
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats = opts.extractStatsFn(net) ;
  stats.num = num ;
  stats.time = time ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  %averageSpeed = (t - lastIndex + batchSize - 1) / time ;
  if t == opts.batchSize + 1
    % compensate for the first iteration, which is an outlier
    adjustTime = 2*batchTime - time ;
    stats.time = time + adjustTime ;
  end

end

if ~isempty(mmap)
  unmap_gradients(mmap) ;
end

prof = [] ;

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)

for p=1:numel(net.params)

  % zero out gradient in dilated regions
  if opts.keepDilatedZeros 
      % only dilated conv in resnet-50 will have 5x5 filter size
      if size(net.params(p).der,1)==5 || size(net.params(p).der,2)==5
          net.params(p).der(2:2:4,:,:,:) = 0;
          net.params(p).der(:,2:2:4,:,:) = 0;
      end
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = ...
          (1 - thisLR) * net.params(p).value + ...
          (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;

    case 'gradient'
      thisDecay = opts.weightDecay * net.params(p).weightDecay ;
      thisLR = state.learningRate * net.params(p).learningRate ;
      state.momentum{p} = opts.momentum * state.momentum{p} ...
        - thisDecay * net.params(p).value ...
        - (1 / batchSize) * net.params(p).der ;
      net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;

    case 'otherwise'
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end


% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end


% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x)(isa(x,'dagnn.HuberLoss')||isa(x,'dagnn.DetLoss')), ...
                   {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end
