import scipy.io


def load_matconvnet_from_file(file_path):
    return scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)


def get_values_for_multi_keys(dictionary, keys):
    """
    Returns values from dictionary for one or more keys. Keys param is a list.
    :param dictionary: input dict
    :param keys: list of keys
    :return: list of values
    """
    if not hasattr(keys, '__getitem__') or isinstance(keys, basestring):
        keys = [keys]
    return [dictionary[k] for k in keys]
