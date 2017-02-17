function create_test_data_fc(case_id)
run(fullfile(fileparts(mfilename('fullpath')), ...
             '..', '..', 'matconvnet','matlab', 'vl_setupnn.m')) ;
if case_id == 1
    var_id = 34;
    var_id2 = 37;
end

[ net, in_img, label, inputs, output_dir ] = load_net( case_id);

net.eval(inputs);

fc6 = net.vars(var_id).value;
fc7 = net.vars(var_id2).value;

save(fullfile(output_dir, 'in_img.mat'), 'in_img')
save(fullfile(output_dir,'f1_reply_fc6.mat'), 'fc6')
save(fullfile(output_dir,'f1_reply_fc7.mat'), 'fc7')
end
