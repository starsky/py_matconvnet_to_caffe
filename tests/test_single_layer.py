# -*- coding: utf-8 -*-

import unittest
import caffe
from os.path import join
import scipy.io
import numpy as np
import py_matconv_to_caffe.matconv_to_caffe
import py_matconv_to_caffe.convert_model_params
import py_matconv_to_caffe.convert_model_weights
from py_matconv_to_caffe.layers import _dagnn_BatchNorm
from py_matconv_to_caffe import utils
from caffe import layers as L


class TestSingleLayer(unittest.TestCase):
    """Basic test cases."""

    @classmethod
    def setUpClass(cls):
        caffe.set_mode_cpu()

    def before(self, test_dir):
        work_dir = join(test_dir, 'workspace')
        py_matconv_to_caffe.matconv_to_caffe.convert_model(join(test_dir, 'net.mat'), work_dir)
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

    def test_batch_norm_layer(self):
        matconv_net = utils.load_matconvnet_from_file('test_data/ucf101-img-resnet-50-split1/net.mat')['net']
        batch_norm_layer_id = 1;
        input_layer = L.DummyData(shape=dict(dim=[10, 64, 112, 112]))
        batch_norm_layer = matconv_net.layers[batch_norm_layer_id]
        lr_params_dic = py_matconv_to_caffe.convert_model_params._create_lr_params_dic(matconv_net.params)
        caffe_layer = _dagnn_BatchNorm([input_layer], batch_norm_layer, utils.get_values_for_multi_keys(lr_params_dic, batch_norm_layer.params))

        n = caffe.NetSpec()
        n.input_layer = input_layer
        layer_name = '%s_INTER%02d' % (batch_norm_layer.name, 0)
        n.__setattr__(layer_name, caffe_layer[0])
        n.__setattr__(batch_norm_layer.name, caffe_layer[1])

        prototxt = str(n.to_proto())
        output_proto_fn = join('test_data/batch_norm_test/workspace', 'net.prototxt')
        with open(output_proto_fn, 'w') as prototxt_file:
            prototxt_file.write(prototxt)
        net = caffe.Net(output_proto_fn, caffe.TEST)

        matconv_net.layers = [matconv_net.layers[batch_norm_layer_id]]
        net = py_matconv_to_caffe.convert_model_weights.add_params(net, matconv_net)

        input_to_layer = scipy.io.loadmat(join('test_data/batch_norm_test/workspace', 'input_to_batch_norm.mat'),
                                            struct_as_record=False, squeeze_me=True)['input_to_layer']

        input_to_layer = np.rollaxis(input_to_layer, 2, 0)
        input_to_layer = np.rollaxis(input_to_layer, 3, 0)

        net.blobs['input_layer'].data[...] = input_to_layer
        net.forward()

        reference_wo_scale = scipy.io.loadmat(join('test_data/batch_norm_test/workspace', 'activations_wo_scale.mat'),
                                      struct_as_record=False, squeeze_me=True)['activations_wo_scale']
        reference_wo_scale = np.rollaxis(reference_wo_scale, 2, 0)
        reference_wo_scale = np.rollaxis(reference_wo_scale, 3, 0)

        values_wo_scale = net.blobs[layer_name].data

        np.testing.assert_array_almost_equal(reference_wo_scale, values_wo_scale, decimal=1)

    def batch_norm_radnom_data(self):
        matconv_net = utils.load_matconvnet_from_file('test_data/ucf101-img-resnet-50-split1/net.mat')['net']
        batch_norm_layer_id = 1;
        input_layer = L.DummyData(shape=dict(dim=[1, 1, 2, 2]))
        batch_norm_layer = matconv_net.layers[batch_norm_layer_id]
        lr_params_dic = py_matconv_to_caffe.convert_model_params._create_lr_params_dic(matconv_net.params)
        caffe_layer = _dagnn_BatchNorm([input_layer], batch_norm_layer, utils.get_values_for_multi_keys(lr_params_dic, batch_norm_layer.params))

        n = caffe.NetSpec()
        n.input_layer = input_layer
        # layer_name = batch_norm_layer.name
        n.__setattr__(batch_norm_layer.name, caffe_layer)

        prototxt = str(n.to_proto())
        output_proto_fn = join('test_data/batch_norm_test/workspace', 'net.prototxt')
        with open(output_proto_fn, 'w') as prototxt_file:
            prototxt_file.write(prototxt)
        net = caffe.Net(output_proto_fn, caffe.TEST)
        layer_name = 'bn_conv1'
        mu = 1
        sig = 2
        net.params[layer_name][0].data[...] = np.asarray([mu])
        net.params[layer_name][1].data[...] = np.asarray([sig ** 2])
        net.params[layer_name][2].data[...] = 1

        data = np.arange(4).reshape((1,1,2, 2)) * 1.0
        net.blobs['input_layer'].data[...] = data

        data -= mu
        data /= sig

        net.forward()

        np.testing.assert_array_almost_equal(data, net.blobs[layer_name].data, decimal=1)

    # def test2(self):
    #     matconv_net = utils.load_matconvnet_from_file('test_data/ucf101-img-resnet-50-split1/net.mat')['net']
    #     batch_norm_layer_id = 1;
    #     channels = 64
    #     img_x = 112
    #     img_y = 112
    #     batch_size = 10
    #     input_layer = L.DummyData(shape=dict(dim=[batch_size, channels, img_x, img_y]))
    #     batch_norm_layer = matconv_net.layers[batch_norm_layer_id]
    #     lr_params_dic = py_matconv_to_caffe.convert_model_params._create_lr_params_dic(matconv_net.params)
    #     caffe_layer = _dagnn_BatchNorm([input_layer], batch_norm_layer, utils.get_values_for_multi_keys(lr_params_dic, batch_norm_layer.params))
    #
    #     n = caffe.NetSpec()
    #     n.input_layer = input_layer
    #     # layer_name = batch_norm_layer.name
    #     n.__setattr__(batch_norm_layer.name, caffe_layer)
    #
    #     prototxt = str(n.to_proto())
    #     output_proto_fn = join('test_data/batch_norm_test/workspace', 'net.prototxt')
    #     with open(output_proto_fn, 'w') as prototxt_file:
    #         prototxt_file.write(prototxt)
    #     net = caffe.Net(output_proto_fn, caffe.TEST)
    #     layer_name = 'bn_conv1'
    #
    #     matconv_net.layers = [matconv_net.layers[batch_norm_layer_id]]
    #     net = py_matconv_to_caffe.convert_model_weights.add_params(net, matconv_net)
    #     #
    #     # net.params[layer_name][0].data[...] = np.asarray([0] * channels)
    #     # net.params[layer_name][1].data[...] = np.asarray([1] * channels)
    #     # net.params[layer_name][2].data[...] = 1
    #
    #     # data = np.random.randint(0,254,batch_size * img_x * img_y * channels).reshape((batch_size,channels,img_x, img_y))
    #     # net.blobs['input_layer'].data[...] = data
    #     data = scipy.io.loadmat(join('test_data/batch_norm_test/workspace', 'input_to_batch_norm.mat'),
    #                                             struct_as_record=False, squeeze_me=True)['input_to_layer']
    #     data = np.rollaxis(data, 2, 0)
    #     data = np.rollaxis(data, 3, 0)
    #     net.blobs['input_layer'].data[...] = data
    #
    #     net.forward()
    #
    #     reference_wo_scale = scipy.io.loadmat(join('test_data/batch_norm_test/workspace', 'activations_wo_scale.mat'),
    #                                   struct_as_record=False, squeeze_me=True)['activations_wo_scale']
    #     reference_wo_scale = np.rollaxis(reference_wo_scale, 2, 0)
    #     reference_wo_scale = np.rollaxis(reference_wo_scale, 3, 0)
    #
    #     data = scipy.io.loadmat(join('test_data/batch_norm_test/workspace', 'input_to_batch_norm.mat'),
    #                                             struct_as_record=False, squeeze_me=True)['input_to_layer']
    #     data = np.rollaxis(data, 3, 0)
    #     data -= net.params[layer_name][0].data
    #     data /= np.sqrt(net.params[layer_name][1].data)
    #     data = np.rollaxis(data, 3, 1)
    #     net.blobs['input_layer'].data[...] = data
    #
    #     np.testing.assert_array_almost_equal(reference_wo_scale, net.blobs[layer_name].data, decimal=1)#net.blobs[layer_name].data, decimal=1)

    def verify(self, test_dir):
        test_net, test_img, reference_values = self.before(test_dir)

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
            self.assertTrue(np.allclose(values, test_values, atol=1e-2), msg=layer_name)

if __name__ == '__main__':
    unittest.main()
