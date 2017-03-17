function [ output_mat ] = jpeg_to_mat( input_dir, output_file, cropp_size)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    files = dir(fullfile(input_dir,'*.jpg'));
    file_list = {};
    for i=1:length(files)
        file_list{i} = fullfile(input_dir, files(i).name);
    end
    output_mat = vl_imreadjpeg(file_list);
    output_mat = cat(3, output_mat{:});
    output_mat = output_mat(1:cropp_size(1), 1:cropp_size(2), :);
    save(output_file, 'output_mat')
    
end

