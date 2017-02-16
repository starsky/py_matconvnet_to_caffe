function create_test_data_score(case_id)
run(fullfile(fileparts(mfilename('fullpath')), ...
             '..', '..', 'matconvnet','matlab', 'vl_setupnn.m')) ;
if case_id == 1
    layer_id = 39;
end

[ net, in_img, label, inputs, output_dir ] = load_net( case_id);

net.eval(inputs);

score = squeeze(net.vars(layer_id).value);

save(fullfile(output_dir, 'in_img.mat'), 'in_img')
save(fullfile(output_dir,'score.mat.mat'), 'score')
end
