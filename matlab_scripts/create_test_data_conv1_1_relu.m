function create_test_data_conv1_1_relu(case_id)
run(fullfile(fileparts(mfilename('fullpath')), ...
             '..', '..', 'matconvnet','matlab', 'vl_setupnn.m')) ;
if case_id == 1
    var_id = 3;
    var_id2 = 5;
    var_id3 = 32;
end

[ net, in_img, label, inputs, output_dir ] = load_net( case_id);
net.eval(inputs);

conv1_1 = net.vars(var_id).value;
conv1_2 = net.vars(var_id2).value;
pool5 = net.vars(var_id3).value;
save(fullfile(output_dir, 'in_img.mat'), 'in_img')
save(fullfile(output_dir,'f1_reply_conv1_1_relu.mat'), 'conv1_1')
save(fullfile(output_dir,'f1_reply_conv1_2_relu.mat'), 'conv1_2')
save(fullfile(output_dir,'f1_reply_pool5_relu.mat'), 'pool5')
end
