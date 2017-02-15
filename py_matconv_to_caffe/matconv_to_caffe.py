#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
import convert
import utils
import os.path
import os


def main():
    argv = sys.argv
    # Parse arguments
    parser = ArgumentParser(description='Converts model saved MatConvNet to Caffe format.')
    # parser.add_argument('-v', '--verbose', action='store_true', help='run verbose')
    parser.add_argument('matconvnet_file', help='Model in MatConvNet format.')
    parser.add_argument('--output_dir', default='.', help='Directory in which Caffe model definition '
                                                          'and model will be saved. By defult the output files '
                                                          'will be saved in current directory.')
    args = parser.parse_args(argv[1:])

    matconv_net = utils.load_matconvnet_from_file(args.matconvnet_file)
    caffe_netspec = convert.process(matconv_net)
    prototxt = str(caffe_netspec.to_proto())

    output_proto_fn = os.path.join(args.output_dir,
                                   '%s.prototxt' % os.path.splitext(os.path.basename(args.matconvnet_file))[0])
    with open(output_proto_fn, 'w') as prototxt_file:
        prototxt_file.write(prototxt)

if __name__ == '__main__':
    sys.exit(main())
