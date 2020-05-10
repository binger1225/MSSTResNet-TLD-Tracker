

% detect target new center postion
% first use own algorithm, if is not good ,use others
function [new_pos, new_target_sz, bestScaleIndex, bestScore, x_bboxes] = tracker_eval(net, s_z, x_crops, pos, scales, params)


    averageImage = reshape(net.meta.normalization.averageImage,1,1,3);
    
    % reference boxes of templates
    clusters = net.meta.clusters;
    
    x_bboxes = {};
    bboxes = [];
    clusterLength = 16;
    bestScaleIndex = ceil(params.numScale/2);
    bestScore = -Inf;
    idxNum = 5;
    for s = 1: size(x_crops,4)
      img = x_crops(:,:,:,s);
      img = bsxfun(@minus, img, averageImage);

      % eval net
      net.eval({'data', img});

      % collect scores 
      score_cls = gather(net.vars(net.getVarIndex('score_cls')).value);
      score_reg = gather(net.vars(net.getVarIndex('score_reg')).value);
      prob_cls = gather(net.vars(net.getVarIndex('prob_cls')).value);
      clusterLength = size(prob_cls, 1);
      [scores,idx] = sort(prob_cls(:),'descend');
      thisScore = max(scores(:));
      if thisScore > bestScore, bestScore = thisScore; bestScaleIndex = s; end
      % threshold for detection
      idx = idx(1: idxNum);
      [fy,fx,fc] = ind2sub(size(prob_cls), idx);

      % interpret heatmap into bounding boxes 
      cy = (fy-1)* params.clusterStride +params.clusterOffset; 
      cx = (fx-1)* params.clusterStride +params.clusterOffset;
      ch = clusters(fc,4) - clusters(fc,2) + 1;
      cw = clusters(fc,3) - clusters(fc,1) + 1;

      % extract bounding box refinement
      Nt = size(clusters, 1); 
      tx = score_reg(:,:,1:Nt); 
      ty = score_reg(:,:,Nt+1:2*Nt); 
      tw = score_reg(:,:,2*Nt+1:3*Nt); 
      th = score_reg(:,:,3*Nt+1:4*Nt); 

      % refine bounding boxes
      dcx = cw .* tx(idx); 
      dcy = ch .* ty(idx);
      rcx = cx + dcx;
      rcy = cy + dcy;
      rcw = cw .* exp(tw(idx));
      rch = ch .* exp(th(idx));
      scores = score_cls(idx);
      tmp_bboxes = [rcx, rcy, rcw * scales(s), rch  * scales(s)];
      tmp_bboxes = horzcat(tmp_bboxes, scores, repmat(s, size(tmp_bboxes,1), 1));
      bboxes = vertcat(bboxes, tmp_bboxes);
    end
    % sort by scores desc
   [scores,idx2] = sort(bboxes(:,5),'descend');
   bboxes = bboxes(idx2(1: idxNum),:);
   bboxes = mean(bboxes(:,[1:4]),1);
   instance = bboxes([2 1]) - (params.clusterStride * clusterLength /2 + params.clusterOffset);
   instance = instance .* s_z ./ [params.exemplarSize params.exemplarSize];
   new_pos = pos + instance; 
   new_target_sz = bboxes([4 3]) .* s_z ./ [params.exemplarSize params.exemplarSize];
   
end

