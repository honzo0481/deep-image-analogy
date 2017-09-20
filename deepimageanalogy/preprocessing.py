"""Implements the preprocessing steps from the Deep Analogy algorithm."""

import numpy as np

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


def get_feature_pyramids(A, Bp):
    """Feed two images into VGG-19 and extract feature maps from the conv layers.

    Parameters
    ----------
    A: ndarray
        The structural image.
    Bp: ndarray
        The style image.

    Returns
    -------
    FA: dict
        A dictionary containing the feature pyramid of A. FA[1] is the output
        from the VGG-19 block1_conv1 layer, etc.
    FBp: dict
        A dictionary containing the feature pyramid of Bp.

    """
    FA, FBp = {}, {}
    # TODO: input_shape should be a function Parameter.
    #       for now use the input shape from the paper's examples.
    vgg19 = VGG19(include_top=False, input_shape=(448, 448, 3))
    layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    for layer, i in enumerate(layers, 1):
        block = Model(inputs=vgg19.input, outputs=vgg19.get_layer(layer).output)
        FA[i] = block.predict(A)
        FBp[i] = block.predict(Bp)

    return FA, FBp
