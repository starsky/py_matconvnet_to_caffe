vl_setupnn
net = load('../../test_data/ucf101-img-resnet-50-split1/net.mat');
net = net.net;
net_mat = net;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
net.conserveMemory = false;
    
cropp_size = net.meta.normalization.imageSize;
output_dir = '../../test_data/batch_norm_test/workspace';
mkdir(output_dir);

%input_mat = jpeg_to_mat( frames_dir,...
%    fullfile(output_dir, 'net_input.mat'), cropp_size);
batch_norm_layer_id = 2;

input_size = net.meta.normalization.imageSize;
batch_size = 10;
data = single(random('unid',255, input_size(1), input_size(2),...
    input_size(3), batch_size));
labels = random('unid',2, 1, batch_size);

inputs = {'input', data, 'label', labels};
net.eval(inputs);

input_to_layer = net.vars(net.layers(batch_norm_layer_id).inputIndexes).value;
activations = net.vars(net.layers(batch_norm_layer_id).outputIndexes).value;

save(fullfile(output_dir, 'input_to_batch_norm.mat') , 'input_to_layer');
save(fullfile(output_dir, 'activations.mat') , 'activations');

net = load('../../test_data/ucf101-img-resnet-50-split1/net.mat');
net = net.net;
net_mat = net;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
net.conserveMemory = false;

net.params(net.layers(batch_norm_layer_id).paramIndexes(1)).value =...
    (net.params(net.layers(batch_norm_layer_id).paramIndexes(1)).value * 0) + 1;
net.params(net.layers(batch_norm_layer_id).paramIndexes(2)).value =...
    (net.params(net.layers(batch_norm_layer_id).paramIndexes(2)).value * 0);

net.eval(inputs);
activations_wo_scale = net.vars(net.layers(batch_norm_layer_id).outputIndexes).value;
save(fullfile(output_dir, 'activations_wo_scale.mat') , 'activations_wo_scale');
