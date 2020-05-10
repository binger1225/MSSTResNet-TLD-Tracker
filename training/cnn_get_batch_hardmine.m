%  FILE:   cnn_get_batch_hardmine.m
% 
%    This function takes a batch of images (including paths and annotations) and
%    generate input and ground truth that will be fed into the detection
%    network.
% 
%  INPUT:  imgs (image paths of a batch of images)
%          imageSizes (image sizes of the same batch of images)
%          labelRects (ground truth bounding boxes)
% 
%  OUTPUT: images (500x500 random cropped regions)
%          clsmaps (ground truth classification heat map)
%          regmaps (ground truth regression heat map)

function [images, clsmaps, regmaps] = cnn_get_batch_hardmine(imgs, labelRects, varargin)

opts.inputSize = [227, 227] ;
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
% opts.numAugments = 1; % no need
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;

opts.rfs = [];
opts.lossType = [];
opts.clusterType = [];
opts.clusters = [];
opts.posThresh = 0.7;
opts.negThresh = 0.3;
opts.var2idx = [];
opts.varsizes = []; 
opts.recfields = [];
opts.sampleSize = 64;
opts.posFraction = 0.5;
opts.video = '';
opts = vl_argparse(opts, varargin);

inputSize = opts.inputSize;


if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
  opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
  opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

for i = 1:numel(labelRects)
  if ~isempty(labelRects{i})
    labelRects{i} = labelRects{i}(:,1:4);
  end
end


fetch = 1;
% most time spent here
if fetch
  imageCells = cell(numel(imgs));
  for i = 1:numel(imgs)
    % create a buffer with all zeros to fill in
    labelRect = labelRects{i};

    img = imgs{i};
    
    if size(img,3) == 1
      img = cat(3, img, img, img) ;
    end

    % update both image and annotation
    imageCells{i} = img;
    labelRects{i} = labelRect;
  end
end

%% decide to use clusters or subclusters 
centers = opts.clusters;

% define grids 
if ~isempty(opts.recfields) && ~isempty(opts.varsizes)
  rf = opts.recfields(opts.var2idx('score_cls'));
  ofx = rf.offset(2);
  ofy = rf.offset(1);
  stx = rf.stride(2);
  sty = rf.stride(1);

  %
  varsize = opts.varsizes{opts.var2idx('score_cls')};
  vsx = varsize(2); 
  vsy = varsize(1);

  % 
  [coarse_xx,coarse_yy] = meshgrid(ofx+(0:vsx-1)*stx, ofy+(0:vsy-1)*sty);
  
  nt = size(centers,1);
  dx1 = reshape(centers(:,1),1,1,nt); 
  dy1 = reshape(centers(:,2),1,1,nt); 
  dx2 = reshape(centers(:,3),1,1,nt); 
  dy2 = reshape(centers(:,4),1,1,nt); 
  
  coarse_xx1 = bsxfun(@plus, coarse_xx, dx1);
  coarse_yy1 = bsxfun(@plus, coarse_yy, dy1);
  coarse_xx2 = bsxfun(@plus, coarse_xx, dx2);
  coarse_yy2 = bsxfun(@plus, coarse_yy, dy2);

  % paste-related viomask
  pad_viomasks = cell(1, numel(imageCells));
  for i = 1:numel(imageCells)
    pasteBoxes = [1 1 inputSize(2) inputSize(1)];
    padx1 = coarse_xx1 < pasteBoxes(1); 
    pady1 = coarse_yy1 < pasteBoxes(2); 
    padx2 = coarse_xx2 > pasteBoxes(3); 
    pady2 = coarse_yy2 > pasteBoxes(4);
    pad_viomasks{i} = padx1 | pady1 | padx2 | pady2; 
  end
end

% init inputs
images = zeros(inputSize(1), inputSize(2), 3, ...
               numel(imgs), 'single') ;

% init targets
if isempty(labelRects) || isempty(opts.varsizes)
  clsmaps = []; 
  regmaps = []; 
else
  clsmaps = -ones(vsy, vsx, nt, numel(imgs), 'single');
  regmaps = zeros(vsy, vsx, nt*4, numel(imgs), 'single');
end

% enumerate
for i=1:numel(imgs)
  % acquire image
  imt = imageCells{i} ;
  if size(imt,3) == 1
    imt = cat(3, imt, imt, imt) ;
  end
  
  % acquire labelRects 
  labelRect = [];
  if ~isempty(labelRects)
    labelRect = labelRects{i};
  end
  


  if ~isempty(opts.averageImage)
    offset = opts.averageImage ;
    if ~isempty(opts.rgbVariance)
      offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3));
    end
     images(:,:,:,i) = bsxfun(@minus, imt, offset) ;
  else
     images(:,:,:,i) = imt;
  end

  
  %% NOTE if this holds, it means we are getting average images 
  if isempty(opts.recfields) || isempty(opts.varsizes)
    continue;
  end

  % initialize IOU matrix 
  iou = [];
  
  ng = size(labelRect, 1);
  if ng > 0
    gx1 = labelRect(:,1);
    gy1 = labelRect(:,2);
    gx2 = labelRect(:,3);
    gy2 = labelRect(:,4);

    iou = compute_dense_overlap(ofx,ofy,stx,sty,vsx,vsy,...
                                dx1,dy1,dx2,dy2,...
                                gx1,gy1,gx2,gy2,...
                                1,1);
    
    fxx1 = reshape(labelRect(:,1),1,1,1,ng);
    fyy1 = reshape(labelRect(:,2),1,1,1,ng); 
    fxx2 = reshape(labelRect(:,3),1,1,1,ng); 
    fyy2 = reshape(labelRect(:,4),1,1,1,ng);

    % compute reg targets
    dhh = dy2-dy1+1;
    dww = dx2-dx1+1;
    fcx = (fxx1 + fxx2)/2; 
    fcy = (fyy1 + fyy2)/2;
    tx = bsxfun(@rdivide, bsxfun(@minus,fcx,coarse_xx), dww);
    ty = bsxfun(@rdivide, bsxfun(@minus,fcy,coarse_yy), dhh);
    fhh = fyy2-fyy1+1;
    fww = fxx2-fxx1+1;
    tw = log(bsxfun(@rdivide, fww, dww)); 
    th = log(bsxfun(@rdivide, fhh, dhh));
  end

  if ~isempty(iou)
    iou = iou + 1e-6*rand(size(iou));
  end

  clsmap = -ones(vsy, vsx, nt, 'single');
  regmap = zeros(vsy, vsx, 4*nt, 'single');

  if ng > 0
    [best_iou,best_face_per_loc] = max(iou, [], 4);
    regidx = sub2ind([vsy*vsx*nt, ng], (1:vsy*vsx*nt)', ...
                     best_face_per_loc(:));
    tx = reshape(tx(regidx), vsy, vsx, nt);
    ty = reshape(ty(regidx), vsy, vsx, nt); 
    tw = repmat(tw, vsy, vsx, 1, 1);
    tw = reshape(tw(regidx), vsy, vsx, nt);
    th = repmat(th, vsy, vsx, 1, 1);
    th = reshape(th(regidx), vsy, vsx, nt);
    regmap = cat(3, tx, ty, tw, th);
    clsmap = max(clsmap, (best_iou>=opts.posThresh)*2-1);
    gray = -ones(size(clsmap));
    gray(opts.negThresh <= best_iou & best_iou < opts.posThresh) = 0;
    clsmap = max(clsmap, gray);
  end

  nonneg_border = (pad_viomasks{i} & clsmap~=-1);
  
  clsmap(nonneg_border) = 0;
  regmap(nonneg_border) = 0;

  clsmaps(:,:,:,i) = clsmap;
  regmaps(:,:,:,i) = regmap;

end
end


