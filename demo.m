
%% tacker demo

clc,clear all,close all;
clear mex;

startup;

datasets={
    struct('name','OTB','basePath','dataset\OTB')
    struct('name','VOT','basePath','dataset\VOT')
    struct('name','UAV20L','basePath','dataset\UAV20L')
};


[video, base_path, dataset] = choose_video(datasets);

% conf = genConfig('OTB','Biker');
% conf = genConfig('VOT','matrix');
% conf = genConfig('UAV20L','person14');

conf = load_video_info(base_path, video, dataset);

% params init
switch(conf.dataset)
    case {'OTB','VOT'}
        params = tracker_init();
        
    case 'UAV20L'
        params = tracker_init_UAV20L();
        
end

% run tracker
[positions, time] = tracker_run(conf.imgList, conf.gt(1,:), conf.video, params);

% print precision and overlap
[precisions, overlaps] = precision_overlap_plot_box(positions, conf.gt, conf.video, params.show_plots);
 fps = numel(conf.imgList) / time;
 fprintf('MSSTResNetTLD tracker:  %12s - Precision (20px):% 1.3f, Overlap (0.5): % 1.3f, FPS:% 4.2f\n', conf.video, precisions(21), overlaps(11), fps);






  
 