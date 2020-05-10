function net = cnn_add_loss_fcn8s_resnet50_simple(opts, net)

%%
if opts.freezeResNet,
    for i = 1:numel(net.params)
        net.params(i).learningRate = 0;
    end
end

%% 
N = opts.clusterNum;
skipLRMultipliers = opts.skipLRMult;
learningRates = skipLRMultipliers;

%% remove prob
if ~isnan(net.getLayerIndex('prob'))
    net.removeLayer('prob');
end

% 
names = {};
for i = 1:numel(net.layers)
    if ~isempty(strfind(net.layers(i).name,'res5')) || ...
            ~isempty(strfind(net.layers(i).name, 'bn5'))
        names{end+1} = net.layers(i).name; 
    end
end
names{end+1} = 'pool5'; 
names{end+1} = 'fc1000';

for i = 1:numel(names)
    net.removeLayer(names{i});
end


%% add predictors on 'res4b22x'
filter = zeros(1,1,1024,5*N,'single');
bias = zeros(1,5*N,'single');
cblk = dagnn.Conv('size',size(filter),'stride',1,'pad',0);
net.addLayer('score_res4', cblk, 'res4fx', 'score_res4', ...
             {'score_res4_filter', 'score_res4_bias'});
fidx = net.getParamIndex('score_res4_filter'); 
bidx = net.getParamIndex('score_res4_bias'); 
net.params(fidx).value = filter;
net.params(fidx).learningRate = learningRates(2);
net.params(bidx).value = bias; 
net.params(bidx).learningRate = learningRates(2);

%% add upsampling 
filter = single(bilinear_u(4, 1, 5*N));

ctblk = dagnn.ConvTranspose('size', size(filter), 'upsample', 2, ...
                                'crop', [1,1,1,1], 'hasBias', false);

% define bilinear interpolation filter (fixed weights) 
net.addLayer('score4', ctblk, 'score_res4', 'score4', 'score4f');
fidx = net.getParamIndex('score4f');
net.params(fidx).value = filter;
net.params(fidx).learningRate = 0;

%% add predictors on 'res3dx'
filter = zeros(1,1,512,5*N,'single');
bias = zeros(1,5*N,'single');
cblk = dagnn.Conv('size',size(filter),'stride',1,'pad',0);
net.addLayer('score_res3', cblk, 'res3dx', 'score_res3', ...
             {'score_res3_filter', 'score_res3_bias'});
fidx = net.getParamIndex('score_res3_filter'); 
bidx = net.getParamIndex('score_res3_bias'); 
net.params(fidx).value = filter;
net.params(fidx).learningRate = learningRates(3);
net.params(bidx).value = bias; 
net.params(bidx).learningRate = learningRates(3);

% sum 
net.addLayer('fusex',dagnn.Sum(),{'score_res3', 'score4'}, ...
             'score_final');
         
net.addLayer('split', dagnn.Split('childIds', {1:N, N+1:5*N}), ...
             'score_final', {'score_cls', 'score_reg'});
   
%% add sigmoid funciton for thresh detect here£¬ convenient for update Net
net.addLayer('prob_cls', dagnn.Sigmoid(), 'score_cls', 'prob_cls');

% only use customized loss when we have variable sample size
net.addLayer('loss_cls', dagnn.Loss('loss', 'logistic'), ...
             {'score_cls', 'label_cls'}, 'loss_cls');
net.addLayer('loss_reg', dagnn.HuberLoss(), ...
             {'score_reg', 'label_reg', 'label_cls'}, 'loss_reg');

