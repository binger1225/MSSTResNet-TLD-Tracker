function config = load_video_info(base_path, video, dataset)

% GENCONFIG
% Generate a configuration of a sequence
% 
% INPUT:
%   dataset - The name of dataset ('otb','vot2013','vot2014','vot2015')
%   video - The name of a sequence in the given dataset
%
% OUTPUT:
%   config - The configuration of the given sequence
%
% Hyeonseob Nam, 2015
% 


%full path to the video's files
if base_path(end) ~= '/' && base_path(end) ~= '\',
base_path(end+1) = '/';
end

    
config.dataset = dataset;
config.video = video;

switch(dataset)
    case {'OTB'}
        config.imgDir = fullfile(base_path, config.video, 'img');

        if(~exist(config.imgDir,'dir'))
            error('%s does not exist!!',config.imgDir);
        end
        
        % parse img list
        config.imgList = parseImg(config.imgDir);
        switch(config.video)
            case 'David'
                config.imgList = config.imgList(300:end);
            case 'Tiger1'
                config.imgList = config.imgList(6:end);
        end
        
        % load gt
        gtPath = fullfile(base_path, config.video, 'groundtruth_rect.txt');
        
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        gt = importdata(gtPath);
        switch(config.video)
            case 'Tiger1'
                gt = gt(6:end,:);
            case {'Board','Twinnings'}
                gt = gt(1:end-1,:);
        end
        config.gt = gt;
        
        nFrames = min(length(config.imgList), size(config.gt,1));
        config.imgList = config.imgList(1:nFrames);
        config.gt = config.gt(1:nFrames,:);
        
   case {'VOT'}
    
        % img path
        config.imgDir = fullfile(base_path, config.video);
        if(~exist(config.imgDir,'dir'))
            error('%s does not exist!!',config.imgDir);
        end
        
        % parse img list
        images = dir(fullfile(config.imgDir,'*.jpg'));
        images = {images.name}';
        images = cellfun(@(x) fullfile(config.imgDir,x), images, 'UniformOutput', false);
        config.imgList = images;
        
        % gt path
        gtPath = fullfile(base_path, config.video, 'groundtruth.txt');
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        % parse gt
        gt = importdata(gtPath);
        if size(gt,2) >= 6
            x = gt(:,1:2:end);
            y = gt(:,2:2:end);
            gt = [min(x,[],2), min(y,[],2), max(x,[],2) - min(x,[],2), max(y,[],2) - min(y,[],2)];
        end
        config.gt = gt;
        
        nFrames = min(length(config.imgList), size(config.gt,1));
        config.imgList = config.imgList(1:nFrames);
        config.gt = config.gt(1:nFrames,:);
        
   case {'UAV20L'}
        
        % img path
        config.imgDir = fullfile(base_path, config.video);
        if(~exist(config.imgDir,'dir'))
            error('%s does not exist!!',config.imgDir);
        end
        
        % parse img list
        images = dir(fullfile(config.imgDir,'*.jpg'));
        images = {images.name}';
        images = cellfun(@(x) fullfile(config.imgDir,x), images, 'UniformOutput', false);
        config.imgList = images;
        
        % gt path
        gtPath = fullfile(base_path, [config.video '.txt']);
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        % parse gt
        gt = importdata(gtPath);
        if size(gt,2) >= 6
            x = gt(:,1:2:end);
            y = gt(:,2:2:end);
            gt = [min(x,[],2), min(y,[],2), max(x,[],2) - min(x,[],2), max(y,[],2) - min(y,[],2)];
        end
        config.gt = gt;
        
        nFrames = min(length(config.imgList), size(config.gt,1));
        config.imgList = config.imgList(1:nFrames);
        config.gt = config.gt(1:nFrames,:);  
        
    case {'new_dataset'}
        % configure new sequence

end
 
end

