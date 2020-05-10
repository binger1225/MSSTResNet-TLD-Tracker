%  FILE:   cnn_load_pretrain.m
%
%    This function takes an input network and fill its parameter weights based
%    on a pretrained network. Parameters are matched based on names. If target
%    network is empty, then use the structure of pretrained network. It happens
%    when cnn_init.m fails to initialize the model structure.
%  
%  INPUT:  net (target network) 
%          prepath (path to a network with pretrained weights)
% 
%  OUTPUT: net (target network with pretrained weights)


function net = cnn_load_pretrain(prepath)

% convert pretrained network to DagNN (easy indexing) 
prenet_ = load(prepath);
if isfield(prenet_, 'net')
    prenet_ = prenet_.net;
end

if isfield(prenet_, 'params')
    prenet = dagnn.DagNN.loadobj(prenet_);
else
    prenet = dagnn.DagNN.fromSimpleNN(prenet_);
end

clear prenet_;

net = prenet;

if isempty(net.getLayerIndex('drop6'))
    net.addLayer('drop6', dagnn.DropOut('rate', 0.5), 'fc6x', 'fc6xd');
    net.setLayerInputs('fc7', {'fc6xd'});
end

if isempty(net.getLayerIndex('drop7'))
    net.addLayer('drop7', dagnn.DropOut('rate', 0.5), 'fc7x', 'fc7xd');
    net.setLayerInputs('score_fr', {'fc7xd'});
end

% remove average image 
net.meta.normalization.averageImage = []; 

for i = 1:numel(net.layers)
    if isa(net.layers(i).block, 'dagnn.BatchNorm')
        midx = net.getParamIndex(net.layers(i).params{1}); % multiplier
        bidx = net.getParamIndex(net.layers(i).params{2}); % bias
        % vectorize 
        net.params(midx).value = net.params(midx).value(:);
        net.params(bidx).value = net.params(bidx).value(:);
    end
end