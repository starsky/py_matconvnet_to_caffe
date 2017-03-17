#test_cases = a.o b.o
test_case_vgg_img = test_data/ucf101-img-vgg16-split1/
test_case_vgg_flow = test_data//ucf101-TVL1flow-vgg16-split1/
test_case_resnet50_img = test_data/ucf101-img-resnet-50-split1/
matconv_net_dir = ../matconvnet/matlab/

init:
	pip install -r requirements.txt

test: prep_data
	nosetests tests

prep_data:
	py_matconv_to_caffe/matlab_scripts/py_matconv2caffe_prepare_test_data $(test_case_vgg_img)/net.mat $(test_case_vgg_img)/test_frames $(test_case_vgg_img)/workspace $(matconv_net_dir)
	py_matconv_to_caffe/matlab_scripts/py_matconv2caffe_prepare_test_data $(test_case_vgg_flow)/net.mat $(test_case_vgg_flow)/test_frames $(test_case_vgg_flow)/workspace $(matconv_net_dir)
	py_matconv_to_caffe/matlab_scripts/py_matconv2caffe_prepare_test_data $(test_case_resnet50_img)/net.mat $(test_case_resnet50_img)/test_frames $(test_case_resnet50_img)/workspace $(matconv_net_dir)
