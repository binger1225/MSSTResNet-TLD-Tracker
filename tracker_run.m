% 
% tracker main function
%
function [positions, time] = tracker_run(img_files, region, video, varargin)

    
    target_sz  = region([4 3]);
    pos = region([2 1]) + target_sz/2;

    % train net params
    params.train.inputSize = [127 127];
    params.train.learningRate = [1e-4*ones(1,30)];
    params.train.modelType = 'resnet-50-simple'; 
    % params.train.modelType = 'resnet-101-simple';
    params.train.networkType = 'dagnn';
    params.train.numEpochs = 50;
    params.train.clusterNum = 67;
    params.train.sampleSize = 256;
    params.train.batchSize = 12;
    params.train.posFraction = 0.5;
    params.train.posThresh = 0.7;
    params.train.negThresh = 0.3;
    params.train.border = [0, 0];
    params.train.freezeResNet = false;
    params.train.skipLRMult = [0 1 0.1];
    params.train.gpus = [];


    params.update = params.train;
    params.update.numEpochs = 10;

    % tracker params
    params.updateNetFrame = 10;
    params.contextAmount = 0.5; % context amount for the exemplar
    params.exemplarSize = 127; % input z size
    params.numScale = 3;
    params.scaleStep = 1.05;
    params.clusterScale = 1.05;
    params.clusterStride = 8;
    params.resetThreshMin = 0;
    params.clusterOffset = -1;
    params.subMean = false;
    params.show_visualization = 0;
    params.show_plots =  0;
    
    switch params.train.modelType
      case 'resnet-50-simple'
         params.train.pretrainModelPath = 'models/imagenet-resnet-50-dag.mat';
      case 'resnet-101-simple'
         params.train.pretrainModelPath = 'models/imagenet-resnet-101-dag.mat';
    end
    

    
     % Overwrite default parameters with varargin
    params = vl_argparse(params, varargin);

    % load net
    net = cnn_load_pretrain(params.train.pretrainModelPath);

    net.meta.inputSize = params.train.inputSize;
    net.meta.normalization.inputSize = params.train.inputSize;
    net.meta.normalization.border = params.train.border;
    net.meta.augmentation.transformation = 'none'; 
    net.meta.augmentation.rgbVariance = [];


    %% compute image stats
    imageStatsPath = 'models/imageStats.mat';
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    net.meta.normalization.averageImage = rgbMean ;
    [v,d] = eig(rgbCovariance) ;
    net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
    clear v d ;

    %% add predictors/losses
    switch params.train.modelType
      case 'resnet-50-simple'
        net = cnn_add_loss_fcn8s_resnet50_simple(params.train, net);
      case 'resnet-101-simple'
        net = cnn_add_loss_fcn8s_resnet101_simple(params.train, net);
      otherwise
        error(sprintf('Not Implemented: model type %s', params.train.modelType));
    end

    %% compute receptive fields and canonical variable sizes 
    var2idx = containers.Map;
    for i = 1:numel(net.vars)
        var2idx(net.vars(i).name) = i;
    end
    net.meta.var2idx = var2idx;

    sz_ = params.train.inputSize;
    net.meta.varsizes = net.getVarSizes({'data',[sz_,3,1]});
    net.meta.recfields = net.getVarReceptiveFields('data');

    %% configure sampling hyperparameters
    net.meta.sampleSize = params.train.sampleSize;
    net.meta.posFraction = params.train.posFraction;
    net.meta.posThresh = params.train.posThresh;
    net.meta.negThresh = params.train.negThresh;

     
    startFrame = 1;
    im = single(imread(img_files{1}));
    % if grayscale repeat one channel to match filters size
	if(size(im,3)==1), im = cat(3,im,im,im); end
     % get avg for padding
    avgChans = net.meta.normalization.averageImage;
  
    im_sz = size(im);
    s_z = get_search_window(target_sz, im_sz, params);
    
    total_crop = {};
    total_label = {};

    [img_crop, ~, img_label] = get_subwindow(im, pos, target_sz, [params.exemplarSize params.exemplarSize], s_z, avgChans);
    if params.subMean
        img_crop = bsxfun(@minus, img_crop, reshape(avgChans, [1 1 3]));
    end
    
    % load cluster
    clusters = tracker_cluster_rects(img_label([3 4]), params.clusterScale, params.train.clusterNum);
    net.meta.clusters = clusters;
    
    scales = (params.scaleStep .^ ((ceil(params.numScale/2)-params.numScale) : floor(params.numScale/2)));

    total_crop{1} = img_crop;
    total_label{1} = img_label;


    imdb = cnn_setup_imdb(total_crop, total_label);
    fprintf('  training Net...\n');
    [net, info] = cnn_tracker_train(net, imdb, params.train, 'video', video);

    % Create video interface for visualization
    if(params.show_visualization)
        update_visualization = show_video(img_files, video);
    end
    
    positions = zeros(numel(img_files), 4);
    tic;
    
    nFrames = numel(img_files);
    for frame = 1: nFrames
        if frame>startFrame
            fprintf('Processing frame %d/%d... \n', frame, nFrames);
             
            im = single(imread(img_files{frame}));
            if(size(im,3)==1), im = cat(3,im,im,im); end

            
            for aa = 1:2

                scaledInstance = scales' * s_z;
                img_crops = make_scale_pyramid(im, pos, scaledInstance, params.exemplarSize, avgChans, params);
                [predict_pos, predict_target_sz, bestScaleIndex, bestScore, predict_bboxes] = tracker_eval(net, s_z, img_crops, pos, scales, params);

                if frame ==2
                    bestScore = max(bestScore, 0.75);
                    params.prob_thresh = bestScore;
                    fprintf('set prob_thresh = %.4f \n', params.prob_thresh);
                end
                
                if aa==2 && bestScore < params.prob_thresh
                    params.prob_thresh = max(bestScore,params.resetThreshMin);
                    fprintf('reset prob_thresh = %.4f \n', params.prob_thresh);
                end
                
                
                if (bestScore<params.prob_thresh && numel(total_crop)>0)

                    total_crop = total_crop(max(1,end-params.updateNetFrame+1):end);
                    total_label = total_label(max(1,end-params.updateNetFrame+1):end);
                    imdb = cnn_setup_imdb(total_crop, total_label);
                    fprintf('  update Net...\n');
                    [net, info] = cnn_tracker_update(net, imdb, params.update);

                    total_crop = {};
                    total_label = {};

                    if bestScore<params.prob_thresh
                        % prediction new location again
                        continue;
                    end
                end
                

                
                if(bestScore>=params.prob_thresh)
                    pos = double(predict_pos);
                    pos(1) = clamp(pos(1),1,size(im,1));
                    pos(2) = clamp(pos(2),1,size(im,2));
                    target_sz = double(predict_target_sz);
                    target_sz(1) = clamp(target_sz(1),1,size(im,1));
                    target_sz(2) = clamp(target_sz(2),1,size(im,2));
                
                    if frame~=numel(img_files)
                        s_z = get_search_window(target_sz, im_sz, params);
                        [img_crop, ~, img_label] = get_subwindow(im, pos, target_sz, [params.exemplarSize params.exemplarSize], s_z, avgChans);
                        if params.subMean
                            img_crop = bsxfun(@minus, img_crop, reshape(avgChans, [1 1 3]));
                        end

                         total_crop{end+1} = img_crop;
                         total_label{end+1} = img_label;
                    end
                end
                   
                break; 
            
            end
        end
        
        
        targetLoc = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        positions(frame, :) = targetLoc;
        if params.show_visualization,
            stop = update_visualization(frame, targetLoc);
            if stop, break, end  %user pressed Esc, stop early
            drawnow
        end

    end
    
    time = toc;
    
    
    
    
end


function y = clamp(x, lb, ub)
% Clamp the value using lowerBound and upperBound

y = max(x, lb);
y = min(y, ub);

end