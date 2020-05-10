function C  = tracker_cluster_rects(rects, scaleStep, N, vis)
if nargin < 5
  vis = 0;
end

%% centralize 
scales = (scaleStep .^ ((ceil(N/2)-N) : floor(N/2)));
scales = sort(scales,'ascend');
rects = round(scales' * rects);
hs = rects(:,1);
ws = rects(:,2);
rects = [-(ws-1)/2, -(hs-1)/2, (ws-1)/2, (hs-1)/2];
C = rects;
%% reorder clusters based on bounding box areas
[~,I] = sort(C(:,3).*C(:,4),'descend');
C = C(I,:);

if ~vis, return; end
subplot = @(m,n,k) subtightplot(m,n,k,[0.1,0.1]);
clf; 
[SI,SJ] = factorize(N);
for i = 1:N
  subplot(SI,SJ,i);
  plotBoxes(C(i,1),C(i,2),C(i,3)-C(i,1)+1,C(i,4)-C(i,2)+1,rand(1,3),0.5);
  title(num2str(i));
  axis([-250,250,-250,250]);
end
end
