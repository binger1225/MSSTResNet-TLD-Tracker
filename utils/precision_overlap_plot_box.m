function [precisions, overlaps] = precision_overlap_plot_box(positions, ground_truth, title, show_plots)
%PRECISION_PLOT
%   Calculates precision for a series of distance thresholds (percentage of
%   frames where the distance to the ground truth is within the threshold).
%   The results are shown in a new figure if SHOW is true.
%
%   Accepts positions and ground truth as Nx2 matrices (for N frames), and
%   a title string.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	
% 	max_threshold = 50;  %used for graphs in the paper
	thresholdSetOverlap = 0:0.05:1;
    thresholdSetError = 0:50;
    
    idxNum = 1;
    precisions = zeros(idxNum, length(thresholdSetError));
    overlaps = zeros(idxNum, length(thresholdSetOverlap));
	
	
    % 第1帧不需要判断错误率和成功率
    positions(1,:) = ground_truth(1,:);
    
    %% 这里一定要切换成"中心位置"
    center = [positions(:,1)+(positions(:,3)-1)/2 positions(:,2)+(positions(:,4)-1)/2];
    centerGT = [ground_truth(:,1)+(ground_truth(:,3)-1)/2 ground_truth(:,2)+(ground_truth(:,4)-1)/2];
% 	if size(center,1) ~= size(centerGT,1),
% % 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
% 		
% 		%just ignore any extra frames, in either results or ground truth
% 		n = min(size(center,1), size(centerGT,1));
% 		center(n+1:end,:) = [];
% 		centerGT(n+1:end,:) = [];
%     end
	errCenter = sqrt(sum(((center - centerGT).^2),2));

    index = ground_truth>0;
    idx=(sum(index,2)==4);
    % errCoverage = calcRectInt(rectMat(1:seq_length,:),rect_anno(1:seq_length,:));
    tmp = calcRectInt(positions(idx,:),ground_truth(idx,:));

    % 错误覆盖率
    errCoverage=-ones(length(idx),1);
    errCoverage(idx) = tmp;
    errCenter(~idx)=-1;

   
	%calculate distances to ground truth over all frames
% 	distances = sqrt((center(:,1) - centerGT(:,1)).^2 + ...
% 				 	 (center(:,2) - centerGT(:,2)).^2);
% 	distances(isnan(distances)) = [];


    for tIdx=1:length(thresholdSetOverlap)
        overlaps(idxNum, tIdx) = sum(errCoverage >thresholdSetOverlap(tIdx)) / numel(errCoverage);
    end


    for tIdx=1:length(thresholdSetError)
        precisions(idxNum, tIdx) = sum(errCenter <= thresholdSetError(tIdx)) / numel(errCenter);
    end
            
            
	%compute precisions
% 	for p = 1:max_threshold,
% 		precisions(p) = nnz(distances <= p) / numel(distances);
%     end
%      for tIdx=1:length(thresholdSetError)
%           precisions(tIdx) = sum(errCenter <= thresholdSetError(tIdx));
%      end
  
    
    %compute success ratio
%     errCoverage = calcRectInt(positions, ground_truth);
%             for tIdx=1:length(thresholdSetOverlap)
%                 overlaps(idx,tIdx) = sum(errCoverage >thresholdSetOverlap(tIdx));
%             end
	
	%plot the precisions
	if show_plots == 1,
		figure('Name',['Precisions - ' title]);
		plot(thresholdSetError, precisions, 'k-', 'LineWidth',2);
		xlabel('Threshold'), ylabel('Precision');
        
        figure('Name',['Overlaps - ' title]);
		plot(thresholdSetOverlap, overlaps, 'k-', 'LineWidth',2);
		xlabel('Threshold'), ylabel('Overlaps');
	end
	
end

