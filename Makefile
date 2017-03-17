#test_cases = a.o b.o
test_case_vgg_img = test_data/ucf101-img-vgg16-split1/
test_case_vgg_flow = test_data//ucf101-TVL1flow-vgg16-split1/

init:
	pip install -r requirements.txt

test: prep_data
	nosetests tests

prep_data:
	matlab_scripts/prepare_test_data.sh $(test_case_vgg_img)/net.mat $(test_case_vgg_img)/test_frames $(test_case_vgg_img)/workspace
	matlab_scripts/prepare_test_data.sh $(test_case_vgg_flow)/net.mat $(test_case_vgg_flow)/test_frames $(test_case_vgg_flow)/workspace
