from PIL import Image
import numpy as np

def modeI2L(img):
    arr = np.array(img).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)
    
class One2Three(object):
    def __call__(self, x):
        return x.repeat(3, 1, 1)

