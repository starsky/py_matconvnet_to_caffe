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
    lr_params_dic = create_learned_params_dic(mcn_net.params)
    for mcn_layer in filter(lambda x: len(x.params) > 0, mcn_net.layers):
        layer_name = mcn_layer.name
        learned_params = get_values_for_multi_keys(lr_params_dic, mcn_layer.params)
        net.params[layer_name][0].data[:, :, :, :] = conver_conv_lerned_filter(learned_params[0]) #np.ones((64,3,3,3)) #learned_params[0]
        if len(learned_params) == 2: #bias exists
            net.params[layer_name][1].data[:] = learned_params[1]
    return net


def conver_conv_lerned_filter(mcn_filter_bank):
    sh = mcn_filter_bank.shape
    if len(sh) == 2:
        ret = np.zeros((sh[-1], sh[-2], 1, 1))
    elif len(sh) == 4:
        ret = np.zeros((sh[-1], sh[-2], sh[-3], sh[-4]))
    else:
        raise ValueError('Did not expect other filter bank size')
    if len(sh) == 2:
        ret[:, :, 0, 0] = np.transpose(mcn_filter_bank) #Not sure about this transpose
    else:
        mcn_filter_bank = np.rollaxis(mcn_filter_bank, 3, 0)
        mcn_filter_bank = np.rollaxis(mcn_filter_bank, 3, 1)
        ret[...] = mcn_filter_bank
    return ret


def create_learned_params_dic(matconv_params_list):
    lr_params_dic = {}
    for mcn_lr_params in matconv_params_list:
        lr_params_dic[mcn_lr_params.name] = mcn_lr_params.value
    return lr_params_dic
