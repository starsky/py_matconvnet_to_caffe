from caffe import layers as L
import caffe
from layers import layer_factory
from utils import get_values_for_multi_keys
"""
This module implements functionality which will convert architecture of the net from matconvnet to
caffe. Including all parameters - number of layers, type of layers, layers params (eg. kernel size,
pad size, stride step etc.). This module also transfers training parameters eg. learning rate for given
weights and weight decay.

Please note that in whole project term "parameters" refers to the setup of network. While term 'weights' refers
to trained/learned model parameters which includes weights of connections between layers and filter weights (in CNN
layers).
"""


def convert_net(matconv_net):
    data_shape = matconv_net.meta.normalization.imageSize[:3]
    data_shape = [1, data_shape[-1], data_shape[0], data_shape[1]]
    data = L.DummyData(shape=dict(dim=data_shape))
    label = L.DummyData(shape=dict(dim=[1]))
    n = caffe.NetSpec()
    n.data = data
    n.label = label

    # dictionary with name of output variable of given layer as a key and layer as a value
    layers_dic = {'input': data, 'label': label}
    # dictionary with layer name as key and dict as a value. Dict contains lr_mult and decay_mult
    # which values are obtained from matconv_net.params
    lr_params_dic = _create_lr_params_dic(matconv_net.params)
    for mcn_layer in matconv_net.layers:
        # searching for input layer based on input/output variable names
        bottom = get_values_for_multi_keys(layers_dic, mcn_layer.inputs)
        layer_name = mcn_layer.name
        top = layer_factory(bottom, mcn_layer, get_values_for_multi_keys(lr_params_dic, mcn_layer.params))
        # add just created layer to dict. Key value is a matconvnet output variable name
        layers_dic[mcn_layer.outputs] = top[-1]
        for i, t in enumerate(top[:-1]):
            n.__setattr__('%s_INTER%02d' % (layer_name, i), top)
        n.__setattr__(layer_name, top[-1])
    return n


def _create_lr_params_dic(matconv_params_list):
    lr_params_dic = {}
    for mcn_lr_params in matconv_params_list:
        lr_params_dic[mcn_lr_params.name] = {'lr_mult': mcn_lr_params.learningRate,
                                             'decay_mult': mcn_lr_params.weightDecay}
    return lr_params_dic
