"""Implements the preprocessing steps from the Deep Analogy algorithm."""

import numpy as np

from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
from typing import Tuple, Dict


def get_feature_pyramids(path_A: str,
                         path_Bp: str,
                         target_size: Tuple) -> Tuple[Dict, Dict]:
    """Feed two images into VGG-19 and extract feature maps from the conv layers.

    Parameters
    ----------
    A: str
        Path to the structural image.
    Bp: str
        Path to the style image.

    Returns
    -------
    FA: dict
        A dictionary containing the feature pyramid of A. FA[1] is the output
        from the VGG-19 block1_conv1 layer, etc.
    FBp: dict
        A dictionary containing the feature pyramid of Bp.

    """
    img_A = image.load_img(path_A, target_size=target_size)
    img_Bp = image.load_img(path_Bp, target_size=target_size)

    A = image.img_to_array(img_A)
    A = np.expand_dims(A, axis=0)
    A = preprocess_input(A)
    Bp = image.img_to_array(img_Bp)
    Bp = np.expand_dims(Bp, axis=0)
    Bp = preprocess_input(Bp)

    FA, FBp = {}, {}
    vgg19 = VGG19(include_top=False, input_shape=target_size)
    layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
              'block4_conv1', 'block5_conv1']

    for layer, i in enumerate(layers, 1):
        block = Model(inputs=vgg19.input,
                      outputs=vgg19.get_layer(layer).output)
        FA[i] = block.predict(A)
        FBp[i] = block.predict(Bp)

    return FA, FBp
