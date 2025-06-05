from skimage.exposure import equalize_adapthist
import numpy as np

def apply_clahe(data: np.ndarray):
    """
    """
    data = data.astype(np.float32, copy=False)
    
    for c in range(data.shape[0]):
        data[c]-= data[c].min()
        data[c] /= np.clip(data[c].max(), a_min=1e-8, a_max=None)
        data[c] = equalize_adapthist(data[c], clip_limit = 0.2)

    return data.astype(np.float32)