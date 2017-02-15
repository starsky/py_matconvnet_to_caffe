import scipy.io


def load_matconvnet_from_file(file_path):
    return scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
