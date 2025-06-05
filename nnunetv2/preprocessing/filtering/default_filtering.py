from skimage.exposure import equalize_adapthist

def apply_clahe(data):
    """
    """
    for c in range(1, data.shape[0]):
        data[c] = equalize_adapthist(data[c], clip_limit = 0.2)

    return data