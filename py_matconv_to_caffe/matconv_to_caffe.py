#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
import convert_model_params
import convert_model_weights
import utils
import os.path
import os
import caffe


def convert_model(matconvnet_file, output_dir, rgb2bgr_layer):
    matconv_net = utils.load_matconvnet_from_file(matconvnet_file)['net']
    # Here we convert net to caffe
    caffe_netspec = convert_model_params.convert_net(matconv_net)
    prototxt = str(caffe_netspec.to_proto())
    output_proto_fn = os.path.join(output_dir,
                                   '%s.prototxt' % os.path.splitext(os.path.basename(matconvnet_file))[0])
    with open(output_proto_fn, 'w') as prototxt_file:
        prototxt_file.write(prototxt)

    net = caffe.Net(output_proto_fn, caffe.TEST)

    # print net.params['conv1_1'][0].data
    # print net.params['conv1_1'][0].data.shape
    net = convert_model_weights.add_trained_weights(net, matconv_net, rgb2bgr_layer=rgb2bgr_layer)
    output_model_fn = os.path.join(output_dir,
                                   '%s.caffemodel' % os.path.splitext(os.path.basename(matconvnet_file))[0])
    net.save(output_model_fn)


def main():
    argv = sys.argv
    # Parse arguments
    parser = ArgumentParser(description='Converts model saved MatConvNet to Caffe format.')
    # parser.add_argument('-v', '--verbose', action='store_true', help='run verbose')
    parser.add_argument('matconvnet_file', help='Model in MatConvNet format (eg. net.mat).')
    parser.add_argument('--output_dir', default='.', help='Directory in which Caffe model definition '
                                                          'and model will be saved. By defult the output files '
                                                          'will be saved in current directory.')
    parser.add_argument('--rgb2bgr_layer', help='Name of the layer which weights will be converted to be in BGR order.'
                                                'Typically it is first convolutional layer. Use only when you know, '
                                                'that net was trained on RGB data cause Caffe usses opencv by default '
                                                'and the input to net will be in BGR.')

    args = parser.parse_args(argv[1:])
    convert_model(args.matconvnet_file, args.output_dir, args.rgb2bgr_layer)


if __name__ == '__main__':
    sys.exit(main())
