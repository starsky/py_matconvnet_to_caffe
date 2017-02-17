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

    def before(self, case_id):
        self.test_data_dir = join('../test_data', 'case%d_wrk' % case_id)
        net = caffe.Net(join(self.test_data_dir, 'ucf101-img-vgg16-split1.prototxt'),
                        join(self.test_data_dir, 'ucf101-img-vgg16-split1.caffemodel'), caffe.TEST)

        # I know it is strange to load img from mat file but I found that cv2.imread and vl_imreadjpeg from
        # matlab returns images with mrse 0.8. This is not a huge difference :) but if we want to check
        # if Conv layers (matlab, caffe) return same output, it i better to feed them with exactly same input
        test_img = scipy.io.loadmat(join(self.test_data_dir,'in_img.mat'),
                                    struct_as_record=False, squeeze_me=True)['in_img']
        return net, test_img

    def test_first_conv_layer_one_filter(self):
        self.first_conv_layer_one_filter(1)

    def test_first_conv_layer_all_filters(self):
        self.first_conv_layer_all_filters(1)

    def test_first_conv_layer__with_relu(self):
        self.first_conv_layer__with_relu(1, 'conv1_1')
        self.first_conv_layer__with_relu(1, 'conv1_2')
        self.first_conv_layer__with_relu(1, 'pool5')

    def test_first_pool_layer(self):
        self.first_pool_layer(1)

    def test_first_fc_layer(self):
        self.first_fc_layer(1, 'fc6')
        self.first_fc_layer(1, 'fc7')

    def first_conv_layer_one_filter(self, case_id):
        net, test_img = self.before(case_id)
        test_net = caffe.Net(join('../test_data', 'case%d' % case_id, 'ucf101-img-vgg16-split1_conv1_1.prototxt'), caffe.TEST)

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

    def first_conv_layer_all_filters(self, case_id):
        net, test_img = self.before(case_id)
        test_net = caffe.Net(join('../test_data', 'case%d' % case_id, 'ucf101-img-vgg16-split1_conv1_1.prototxt'), caffe.TEST)

        test_net.params['conv1_1'][0].data[...] = net.params['conv1_1'][0].data[...]
        test_net.params['conv1_1'][1].data[...] = net.params['conv1_1'][1].data[...]

        # load data to net
        test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
        test_net.blobs['data'].data[...] = test_img
        test_net.forward()

        f1_reply = test_net.blobs['conv1_1'].data[0, :, :, :]
        mat_filter_value = scipy.io.loadmat(join(self.test_data_dir, 'f1_reply_full.mat'),
                                            struct_as_record=False, squeeze_me=True)['f1_reply']

        self.assertTrue(np.allclose(np.rollaxis(mat_filter_value, 2, 0), f1_reply, atol=1e-4))

    def first_conv_layer__with_relu(self, case_id, layer_id):
        test_net, test_img = self.before(case_id)

        # load data to net
        test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
        test_net.blobs['data'].data[...] = test_img
        test_net.forward()

        f1_reply = test_net.blobs[layer_id].data[0, :, :, :]
        mat_filter_value = scipy.io.loadmat(join(self.test_data_dir, 'f1_reply_%s_relu.mat' % layer_id),
                                            struct_as_record=False, squeeze_me=True)[layer_id]

        self.assertTrue(np.allclose(np.rollaxis(mat_filter_value, 2, 0), f1_reply, atol=1e-4))

    def first_pool_layer(self, case_id):
        test_net, test_img = self.before(case_id)

        # load data to net
        test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
        test_net.blobs['data'].data[...] = test_img
        test_net.forward()

        f1_reply = test_net.blobs['pool1'].data[0, :, :, :]
        mat_filter_value = scipy.io.loadmat(join(self.test_data_dir, 'f1_reply_pool.mat'),
                                            struct_as_record=False, squeeze_me=True)['f1_reply']

        self.assertTrue(np.allclose(np.rollaxis(mat_filter_value, 2, 0), f1_reply, atol=1e-4))

    def first_fc_layer(self, case_id, layer_id):
        test_net, test_img = self.before(case_id)

        # load data to net
        test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
        test_net.blobs['data'].data[...] = test_img
        test_net.forward()

        f1_reply = test_net.blobs[layer_id].data[0, :, :, :].squeeze()
        mat_filter_value = scipy.io.loadmat(join(self.test_data_dir, 'f1_reply_%s.mat' % layer_id),
                                            struct_as_record=False, squeeze_me=True)[layer_id]

        self.assertTrue(np.allclose(mat_filter_value, f1_reply, atol=1e-4))
    # def test_score(self):
    #     self.test_data_dir = '../test_data'
    #     test_net = caffe.Net(join(self.test_data_dir, 'ucf101-img-vgg16-split1.prototxt'),
    #                          join(self.test_data_dir, 'ucf101-img-vgg16-split1.caffemodel'), caffe.TEST)
    #
    #     # I know it is strange to load img from mat file but I found that cv2.imread and vl_imreadjpeg from
    #     # matlab returns images with mrse 0.8. This is not a huge difference :) but if we want to check
    #     # if Conv layers (matlab, caffe) return same output, it i better to feed them with exactly same input
    #     test_img = scipy.io.loadmat(join(self.test_data_dir, 'in_img.mat'),
    #                                 struct_as_record=False, squeeze_me=True)['in_img']
    #
    #     # load data to net
    #     test_img = np.rollaxis(test_img, 2, 0)  # to have dim [channelxHxW]
    #     test_net.blobs['data'].data[...] = test_img
    #     scores = test_net.forward()
    #     scores = scores['loss39'].squeeze()
    #
    #     mat_score_value = scipy.io.loadmat(join(self.test_data_dir, 'score.mat'),
    #                                         struct_as_record=False, squeeze_me=True)['score']
    #
    #     self.assertTrue(np.allclose(mat_score_value, scores, atol=1e-4))


if __name__ == '__main__':
    unittest.main()