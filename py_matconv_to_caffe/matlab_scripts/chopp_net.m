function [ ret ] = chopp_net(net, input_mat, output_file, label )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    batch = single(zeros([size(input_mat), 1]));
    batch(:, :, :, 1) = input_mat;
    inputs = {'input', batch, 'label', label};
    
    net.eval(inputs);
    ret = struct();
    
    execution_order = net.getLayerExecutionOrder();
    for i=1:length(net.layers)
        layer = net.getLayer(execution_order(i));
        if isa(layer.block, 'dagnn.Loss')
%            isa(layer.block, 'dagnn.ReLU') ||...
%           isa(layer.block, 'dagnn.Sigmoid')||...
           %We do not want to compare Loss layer in testing
           %ReLu and Sigmoid are already saved in conv layers
            continue
        end
%        if i+1 <= length(net.layers)
%            next_layer = net.getLayer(execution_order(i+1));
%        else
%            next_layer = {};
%        end
        layer_name = layer.name;
%        layer = switch_layer(layer, next_layer);
        outputs = layer.outputs;
        if length(outputs) > 1
            fprintf(...
                'Outputs in layer %s have more than one variable - do not know what to do', ...
                layer_name);
            return
        end
        layer_values = net.getVar(outputs(1));
        ret.(layer_name) = layer_values;
    end
    save(output_file, 'ret')

end

%function r = switch_layer(layer, next_layer)
%Because caffe do not save values of linear neuron and non-linearity
%separataly. So in such case conv layer values in caffe are after
%the non-linearity applied. But in matconvnet conv layer and non-linear
%layers are saved separately. To unify the outputs this function
%check if layer is of type Conv (linear) and if yes it returns
%next layer (if next one is non-linear). Then we get similar output
%to caffe.
%    if isempty(next_layer)
%        r = layer;
%        return
%    end
%    if isa(layer.block, 'dagnn.Conv') || isa(layer.block, 'dagnn.BatchNorm')
%        if isa(next_layer.block, 'dagnn.ReLU') ||...
%           isa(next_layer.block, 'dagnn.Sigmoid')     
%            r = switch_layer(next_layer, {});
%        else
%            r = layer;
%        end
%    else
%        r = layer;
%    end
%end

