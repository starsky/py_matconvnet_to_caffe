# -*- coding: utf-8 -*-

from .context import sample

import unittest
import caffe
from os.path import join
import scipy.io
import numpy as np


class TestConv(unittest.TestCase):
    """Basic test cases."""

    @classmethod
    def setUpClass(cls):
        caffe.set_mode_cpu()

    def before(self, work_dir):
        net = caffe.Net(join(work_dir, 'net.prototxt'),
                        join(work_dir, 'net.caffemodel'), caffe.TEST)

        # I know it is strange to load img from mat file but I found that cv2.imread and vl_imreadjpeg from
        # matlab returns images with mrse 0.8. This is not a huge difference :) but if we want to check
        # if Conv layers (matlab, caffe) return same output, it i better to feed them with exactly same input
        test_img = scipy.io.loadmat(join(work_dir, 'net_input.mat'),
                                    struct_as_record=False, squeeze_me=True)['output_mat']
        reference_values = scipy.io.loadmat(join(work_dir, 'activations.mat'),
                                            struct_as_record=False, squeeze_me=True)['ret']
        return net, test_img, reference_values

    def test_verify_net(self):
        test_net, test_img, reference_values = self.before('../test_data/ucf101-img-vgg16-split1/workspace/')

        #load data to net
        test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
        test_net.blobs['data'].data[...] = test_img
        test_net.forward()

        for layer_name in reference_values._fieldnames:
            values = getattr(reference_values, layer_name).value
            test_values = test_net.blobs[layer_name].data[0, :, :, :]
            if len(values.shape) > 1:
                # if values mat is multidimensional we need to roll dimensions to align them
                # to caffe
                values = np.rollaxis(values, 2, 0)
            else:
                # if values is a vector we need to squeeze caffe vector as it holds dimensions with
                # single values
                test_values = test_values.squeeze()
            self.assertTrue(np.allclose(values, test_values, atol=1e-3), msg=layer_name)

if __name__ == '__main__':
    unittest.main()