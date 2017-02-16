# -*- coding: utf-8 -*-

from .context import sample

import unittest
import caffe
from os.path import join
import scipy.io
import numpy as np


class TestConv(unittest.TestCase):
    """Basic test cases."""

    def test_first_conv_layer(self):
        self.test_data_dir = '../test_data'
        net = caffe.Net(join(self.test_data_dir, 'ucf101-img-vgg16-split1.prototxt'),
                        join(self.test_data_dir, 'ucf101-img-vgg16-split1.caffemodel'), caffe.TEST)
        test_net = caffe.Net(join(self.test_data_dir, 'ucf101-img-vgg16-split1_conv1_1.prototxt'), caffe.TEST)

        # I know it is strange to load img from mat file but I found that cv2.imread and vl_imreadjpeg from
        # matlab returns images with mrse 0.8. This is not a huge difference :) but if we want to check
        # if Conv layers (matlab, caffe) return same output, it i better to feed them with exactly same input
        test_img = scipy.io.loadmat(join(self.test_data_dir,'in_img.mat'),
                                    struct_as_record=False, squeeze_me=True)['in_img']

        #some net surgery
        # test_filter = np.transpose(net.params['conv1_1'][0].data[0, 0, :, :])
        test_filter = net.params['conv1_1'][0].data[0, 0, :, :]
        test_net.params['conv1_1'][0].data[0, 0, :, :] = test_filter
        test_net.params['conv1_1'][0].data[1, 1, :, :] = test_filter
        test_net.params['conv1_1'][0].data[2, 2, :, :] = test_filter

        #load data to net
        test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
        test_net.blobs['data'].data[...] = test_img
        test_net.forward()

        f1_reply = test_net.blobs['conv1_1'].data[0, 0, :, :]
        mat_filter_value = scipy.io.loadmat(join(self.test_data_dir,'f1_reply.mat'),
                                            struct_as_record=False, squeeze_me=True)['f1_reply']

        self.assertTrue(np.allclose(mat_filter_value, f1_reply, atol=1e-4))


# def test_first_conv_layer2(self):
#     self.test_data_dir = '../test_data'
#     net = caffe.Net(join(self.test_data_dir, 'ucf101-img-vgg16-split1.prototxt'),
#                     join(self.test_data_dir, 'ucf101-img-vgg16-split1.caffemodel'), caffe.TEST)
#
#     # I know it is strange to load img from mat file but I found that cv2.imread and vl_imreadjpeg from
#     # matlab returns images with mrse 0.8. This is not a huge difference :) but if we want to check
#     # if Conv layers (matlab, caffe) return same output, it i better to feed them with exactly same input
#     test_img = scipy.io.loadmat(join(self.test_data_dir, 'in_img.mat'),
#                                 struct_as_record=False, squeeze_me=True)['in_img']
#
#     # load data to net
#     test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
#     net.blobs['data'].data[...] = test_img
#     net.forward()
#
#     f1_reply = net.blobs['conv1_1'].data[0, :, :, :]
#     mat_filter_value = scipy.io.loadmat(join(self.test_data_dir, 'f1_reply_full.mat'),
#                                         struct_as_record=False, squeeze_me=True)['f1_reply']
#
#     self.assertTrue(np.allclose(mat_filter_value, f1_reply, atol=1e-4))


if __name__ == '__main__':
    unittest.main()