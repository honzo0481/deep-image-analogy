# -*- coding: utf-8 -*-
"""
Deconvolution Step

Only content reconstruction is implemented at the moment.

Models supported are VGG19

https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
"""

import keras.backend as K
import numpy as np
import os

from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from typing import Tuple, List, Dict, Callable


class Deconv(object):

    def __init__(self, model, preprocess_input: Callable, path_A: str,
                 target_size: Tuple=(448, 448)) -> None:
        self.model = model
        self.preprocess_input = preprocess_input
        if not os.path.isfile(path_A):
            raise IOError
        self.img_ncols, self.img_nrows = target_size
        self.img_a = self._preprocess_image(path_A)

    def _preprocess_image(self, path: str) -> np.ndarray:
        """
        Load img and output input tensor with the target size
        """
        img = load_img(path, target_size=(self.img_ncols, self.img_nrows))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input(img)
        return img

    def _deprocess_image(self, x: np.ndarray) -> np.ndarray:
        """
        Remove normalizations and necessary changes to output
        img array
        """
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, self.img_ncols, self.img_nrows))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((self.img_ncols, self.img_nrows, 3))

        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68

        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def _content_loss(self, base: np.ndarray, gen: np.ndarray) -> np.ndarray:
        """
        Loss function for content similiarly between base image and generated
        image

        By minimizing the loss, our generated image will be consistent with
        the base image.
        """
        return K.sum(K.square(gen - base))

    def deconv(self, noise_ratio: float=1.0, content_weight: float=1.0,
               layer: str='block5_conv1', iterations: int=15,
               output_steps: int=5, save_images: bool=True,
               save_dir: str='deconv_results') -> np.ndarray:
        """
        Performs image deconvolution on a layer in the vgg19 model

        :param noise_ratio: float,
        :param content_weight: float,
        :param layer: str,
        :param iterations: int,
        :param output_steps: int,
        :param save_images: bool,
        :param save_dir: str,

        :return array, deprocessed img array after the final iteration of the
        optimization function

        """

        # Get inputs
        a_image = K.variable(self.img_a)
        generated_image = K.placeholder((1, self.img_nrows, self.img_ncols, 3))
        input_tensor = K.concatenate([a_image,
                                      generated_image], axis=0)

        # Get Model
        model = self.model(include_top=False, weights='imagenet',
                           input_tensor=input_tensor)

        feature_maps = dict([(layer.name, layer.output)
                             for layer in model.layers])

        if layer not in feature_maps:
            raise IOError("input layer cannot be found in model")

        # Get desired features
        layer_features = feature_maps[layer]
        a_features = layer_features[0, :, :, :]
        gen_features = layer_features[1, :, :, :]

        # Setup Loss
        loss = K.variable(0.)
        loss += content_weight * self._content_loss(a_features, gen_features)

        # Get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, generated_image)

        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)

        f_outputs = K.function([generated_image], outputs)

        # Init evaluator
        evaluator = Evaluator(f_outputs)

        # Generated random noise from content img
        noise_img = np.random.uniform(-20., 20.,
                                      (1, self.img_ncols, self.img_nrows, 3))
        input_img = noise_ratio * noise_img + (1. - noise_ratio) * self.img_a

        if save_images:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # run scipy-based optimization (L-BFGS) over the pixels of the
        # generated image so as to minimize the content loss
        for i in range(iterations):
            input_img, _, _ = fmin_l_bfgs_b(evaluator.loss,
                                            input_img.flatten(),
                                            fprime=evaluator.grads,
                                            maxfun=20)
            if i % output_steps == 0:
                img = self._deprocess_image(input_img.copy())
                # save current generated image
                if save_images:
                    fname = '{}/from_{}_at_{}.png'.format(save_dir, layer, i)
                    imsave(fname, img)

        return self._deprocess_image(input_img.copy())


class Evaluator(object):
    """
    Evaluator class makes it possible to compute loss and
    gradients in one pass while retrieving them via two separate
    functions, "loss" and "grads". This is done because scipy.optimize
    requires separate functions for loss and gradients, but computing
    them separately would be inefficient
    """

    def __init__(self, f_outputs: Callable) -> None:
        self.loss_value: float = None
        self.grads_values: List = None
        self._f_outputs: Callable = f_outputs

    def loss(self, x: np.ndarray) -> float:
        assert self.loss_value is None
        loss_value, grad_values = self._eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x: np.ndarray) -> List:
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def _eval_loss_and_grads(self, x: np.ndarray,
                             size: Tuple=(448, 448)) -> Tuple[float, List]:
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, size[0], size[1]))
        else:
            x = x.reshape((1, size[0], size[1], 3))
        outs = self._f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
