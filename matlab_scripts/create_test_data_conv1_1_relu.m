function create_test_data_conv1_1_relu(case_id)
run(fullfile(fileparts(mfilename('fullpath')), ...
             '..', '..', 'matconvnet','matlab', 'vl_setupnn.m')) ;
if case_id == 1
    layer_id = 3;
end

[ net, in_img, label, inputs, output_dir ] = load_net( case_id);
net.eval(inputs);

f1_reply = net.vars(layer_id).value;
save(fullfile(output_dir, 'in_img.mat'), 'in_img')
save(fullfile(output_dir,'f1_reply_conv1_1_relu.mat'), 'f1_reply')
end
