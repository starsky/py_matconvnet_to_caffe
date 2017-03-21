import numpy as np
from utils import get_values_for_multi_keys
"""
This module implements functionality which will convert weights form model trained in matconvnet to
caffemodel.

Please note that in whole project term "parameters" refers to the setup of network. While term 'weights' refers
to trained/learned model parameters which includes weights of connections between layers and filter weights (in CNN
layers).
"""


def add_params(net, mcn_net):
    trained_weights_dic = _create_trained_wights_dict(mcn_net.params)

    for mcn_layer in filter(lambda x: len(x.params) > 0, mcn_net.layers):  # Filter layers that have trainable weights
        layer_name = mcn_layer.name
        trained_weights = get_values_for_multi_keys(trained_weights_dic, mcn_layer.params)
        if mcn_layer.type == 'dagnn.BatchNorm':
            _add_weights_BatchNorm(net, layer_name, trained_weights)
        elif mcn_layer.type == 'dagnn.Conv':
            net.params[layer_name][0].data[:, :, :, :] = _convert_trained_convolution_filter(trained_weights[0])
            if len(trained_weights) == 2:  # bias exists
                net.params[layer_name][1].data[:] = trained_weights[1]
        else:
            raise ValueError('Unknown layer weight transfer method for %s' % mcn_layer.type)
    return net


def _add_weights_BatchNorm(net, layer_name, trained_weights):
    normalization_layer_name = '%s_INTER%02d' % (layer_name, 0)
    net.params[normalization_layer_name][0].data[...] = np.copy(trained_weights[2][:, 0])
    # Note that mat_conv_net stores sqrt(var) and caffe stores var in the parameters, that is why
    # we do power(,2) to before storing var value.
    net.params[normalization_layer_name][1].data[...] = np.power(np.copy(trained_weights[2][:, 1]), 2)
    net.params[normalization_layer_name][2].data[...] = 1

    net.params[layer_name][0].data[...] = np.copy(trained_weights[0])
    net.params[layer_name][1].data[...] = np.copy(trained_weights[1])

def _convert_trained_convolution_filter(mcn_filter_bank):
    sh = mcn_filter_bank.shape
    if len(sh) == 2:
        ret = np.zeros((sh[-1], sh[-2], 1, 1))
    elif len(sh) == 4:
        ret = np.zeros((sh[-1], sh[-2], sh[-3], sh[-4]))
    else:
        raise ValueError('Did not expect other filter bank size')
    if len(sh) == 2:
        ret[:, :, 0, 0] = np.transpose(mcn_filter_bank)
    else:
        mcn_filter_bank = np.rollaxis(mcn_filter_bank, 3, 0)
        mcn_filter_bank = np.rollaxis(mcn_filter_bank, 3, 1)
        ret[...] = mcn_filter_bank
    return ret


def _create_trained_wights_dict(matconv_params_list):
    """
    Creates dictionary with layer param name as a key and trained parameters as a value.
    Note that in matconvnet param.value in refers to trained weights.
    :param matconv_params_list:
    :return:
    """
    trained_weights_dic = {}
    for mcn_lr_params in matconv_params_list:
        trained_weights_dic[mcn_lr_params.name] = mcn_lr_params.value
    return trained_weights_dic
