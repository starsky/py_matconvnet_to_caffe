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
            continue
        end
        layer_name = layer.name;
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
