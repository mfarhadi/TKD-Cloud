from dippykit import *

import numpy as np
im = np.array([[  0, 255, 255,   0],
                [255,   0,  64, 128]])
im_vec = im.reshape(-1)


im_encoded, stream_length, symbol_code_dict, symbol_prob_dict = huffman_encode(im_vec)

i=np.unpackbits(im_encoded)
print(symbol_code_dict)