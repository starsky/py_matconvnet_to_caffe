function [ net, in_img, label, inputs, output_dir ] = load_net( case_id)
run(fullfile(fileparts(mfilename('fullpath')), ...
             '..', '..', 'matconvnet','matlab', 'vl_setupnn.m')) ;
if case_id == 1
    fn = '../test_data/case1/ucf101-img-vgg16-split1.mat';
    img_path = '../test_data/case1/';
    output_dir = '../test_data/case1_wrk';
end
         
net = load(fn);
net = net.net;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
net.conserveMemory = false;

tst_img = vl_imreadjpeg({fullfile(img_path, 'frame000001.jpg')});
%tst_img = vl_imreadjpeg({'/home/mkopersk/data/two-streams/ucf101/jpegs_256/v_ApplyEyeMakeup_g04_c03/frame000001.jpg'});
%Uncoment above to test on any other image

in_img = single(zeros(224,224,3,1));
in_img(:,:,:,1) = single(tst_img{1}(1:224,1:224,:));
label = [1];

inputs = {'input', in_img, 'label', label};

end

