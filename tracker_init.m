
% tracker params setting
function params = tracker_init()
 
% net train params
params.train.inputSize = [127 127];
params.train.modelType = 'resnet-50-simple'; 
params.train.numEpochs = 50;
params.train.clusterNum = 67; 
params.train.gpus = [];

% net update params
params.update = params.train;
params.update.numEpochs = 10;
params.updateNetFrame = 10;

% tracking params
params.numScale = 3;
params.scaleStep = 1.05;
params.show_visualization = 1;
params.show_plots = 0;


switch params.train.modelType
  case 'resnet-50-simple'
     params.train.pretrainModelPath = 'models/imagenet-resnet-50-dag.mat';
  case 'resnet-101-simple'
     params.train.pretrainModelPath = 'models/imagenet-resnet-101-dag.mat';
end


end