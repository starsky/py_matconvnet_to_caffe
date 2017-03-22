# Python MatConvNet to Caffe Converter

This project converts the neural network models saved/trained in [MatConvNet](http://www.vlfeat.org/matconvnet/)
to [Caffe](http://caffe.berkeleyvision.org/). The project does not require Matlab to make the conversion, but
if you want to verify your model Matlab will be needed.

## Installation
You can either install the project using `pip install https://github.com/starsky/py_matconvnet_to_caffe.git` or you can clone project directly from the GitHub. After cloning the project you can run tests `make test` to verify that code works fine on your machine. Note that to run the tests you need to have Matlab installed. Before running theteste please download test data from [here](ftp://ftp-sop.inria.fr/members/Michal.Koperski/py_matconvnet_2_caffe_test_data.tar.bz2) and unpack the archive to test_data directory.

## Usage
```
usage: matconv_to_caffe.py [-h] [--output_dir OUTPUT_DIR] matconvnet_file

Converts model saved MatConvNet to Caffe format.

positional arguments:
  matconvnet_file       Model in MatConvNet format (eg. net.mat).

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Directory in which Caffe model definition and model
                        will be saved. By defult the output files will be
                        saved in current directory.

```
## Implemented Layers
I have implemented conversion of the following layers so far:
1. Dagnn.BatchNorm
2. Dagnn.Conv
3. Dagnn.DropOut
4. Dagnn.Loss
5. Dagnn.Pooling
6. Dagnn.ReLU
7. Dagnn.Sum

## Release Notes
With this project I managed to convert vgg-16 net and resnet-50. If you want to convert other neural network
and your are having a trouble - please contact me. I will be happy to help you. 

I wanted to to thank Christoph Feichtenhofer who made the mentioned vgg-16 and resnet-50 models available [here](https://github.com/feichtenhofer/twostreamfusion).

## Contact
In case of any questions please contact me via GitHub.

