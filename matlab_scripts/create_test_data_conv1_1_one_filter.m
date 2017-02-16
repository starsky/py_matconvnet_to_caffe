function create_test_data_conv1_1_one_filter(case_id)

if case_id == 1
    var_id = 2;
    params_id = 1;
    params_bias_id = 2;
end

[ net, in_img, label, inputs, output_dir ] = load_net( case_id);

test_filter = net.params(params_id).value(:,:,1,1);

net.params(params_id).value(:,:,:,:) = 0;
net.params(params_bias_id).value(1:end) = 0;

net.params(params_id).value(:,:,1,1) = test_filter;
net.params(params_id).value(:,:,2,2) = test_filter;
net.params(params_id).value(:,:,3,3) = test_filter;

net.eval(inputs);

f1_reply = net.vars(var_id).value(:,:,1);
save(fullfile(output_dir, 'in_img.mat'), 'in_img')
save(fullfile(output_dir,'f1_reply.mat'), 'f1_reply')