function [ input_mat, activations ] =...
    prepare_test_data_for_net(net_file, frames_dir, output_dir)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%    run(fullfile(fileparts(mfilename('fullpath')), ...
%             '..', '..', 'matconvnet','matlab', 'vl_setupnn.m')) ;
    vl_setupnn
    net = load(net_file);
    net = net.net;
    net = dagnn.DagNN.loadobj(net);
    net.mode = 'test';
    net.conserveMemory = false;
    
    cropp_size = net.meta.normalization.imageSize;
    mkdir(output_dir);
    input_mat = jpeg_to_mat( frames_dir,...
        fullfile(output_dir, 'net_input.mat'), cropp_size);
    
    activations = chopp_net(net, input_mat,...
        fullfile(output_dir, 'activations.mat'), 1 );

end

