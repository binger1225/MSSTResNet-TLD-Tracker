function results = run_MSSTResNetTLD(seq, subA, res_path, bSaveImage)


 clear mex;
 
 startup;

% ================================================================================

conf.video = seq.name;
conf.imgList = seq.s_frames;
conf.gt = seq.init_rect;

params = tracker_init();
 
 % run tracker
[positions, time] = tracker_run(conf.imgList, conf.gt(1,:), conf.video, params);
fps = numel(conf.imgList) / time;

results.type   = 'rect';
results.res    = positions;
results.len    = numel(conf.imgList);
results.fps    = fps;

end

