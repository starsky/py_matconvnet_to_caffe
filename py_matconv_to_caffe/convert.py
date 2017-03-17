from caffe import layers as L
from caffe import params as P
import caffe
import numpy as np
from layers import layer_factory


def process(matconv_net):
    data_shape = matconv_net['net'].meta.normalization.imageSize
    data_shape = [1, data_shape[-1], data_shape[0], data_shape[1]]
    data = L.DummyData(shape=dict(dim=data_shape))
    label = L.DummyData(shape=dict(dim=[1]))
    n = caffe.NetSpec()
    n.data = data
    n.label = label
    layers_dic = {'input': data, 'label': label}
    lr_params_dic = create_lr_params_dic(matconv_net['net'].params)
    for mcn_layer in matconv_net['net'].layers:
        bottom = get_multi_keys(layers_dic, mcn_layer.inputs)
        layer_name = mcn_layer.name
        top = layer_factory(bottom, n.label, mcn_layer, get_multi_keys(lr_params_dic, mcn_layer.params))
        layers_dic[mcn_layer.outputs] = top
        n.__setattr__(layer_name, top)
    return n


def create_lr_params_dic(matconv_params_list):
    lr_params_dic = {}
    for mcn_lr_params in matconv_params_list:
        lr_params_dic[mcn_lr_params.name] = {'lr_mult': mcn_lr_params.learningRate,
                                             'decay_mult': mcn_lr_params.weightDecay}
    return lr_params_dic


def create_learned_params_dic(matconv_params_list):
    lr_params_dic = {}
    for mcn_lr_params in matconv_params_list:
        lr_params_dic[mcn_lr_params.name] = mcn_lr_params.value
    return lr_params_dic




def get_multi_keys(dictionary, keys):
    if not hasattr(keys, '__getitem__') or isinstance(keys, basestring):
        keys = [keys]
    return [dictionary[k] for k in keys]








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


def add_params(net, mcn_net):
    lr_params_dic = create_learned_params_dic(mcn_net.params)
    for mcn_layer in filter(lambda x: len(x.params) > 0, mcn_net.layers):
        layer_name = mcn_layer.name
        learned_params = get_multi_keys(lr_params_dic, mcn_layer.params)
        net.params[layer_name][0].data[:, :, :, :] = conver_conv_lerned_filter(learned_params[0]) #np.ones((64,3,3,3)) #learned_params[0]
        if len(learned_params) == 2: #bias exists
            net.params[layer_name][1].data[:] = learned_params[1]
    return net
