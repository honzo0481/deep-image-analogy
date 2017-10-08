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
from typing import Tuple, List, Callable


class Deconv(object):

    def __init__(self, model, preprocess_input: Callable, path_A: str,
                 path_B: str, target_size: Tuple=(448, 448)) -> None:
        """
        Deconvolution initialization

       :param model: keras model for the VGG architecture
       :param preprocess_input: fn, performs preprocessing operations on img
       :param path_A: str, path to img that holds **content** information
       :param path_B: str, path to img that holds **style** information
       :param target_size: tuple, dimensions of combination pictures
        """
        self.model = model
        self.preprocess_input = preprocess_input
        if not os.path.isfile(path_A):
            raise IOError
        self.img_ncols, self.img_nrows = target_size
        self.img_a = self._preprocess_image(path_A)
        self.img_b = self._preprocess_image(path_B)
        self.features_list = self._get_feature_layers()

    def _get_feature_layers(self) -> List[str]:
        feature_layers = ['block1_conv1', 'block2_conv1',
                          'block3_conv1', 'block4_conv1',
                          'block5_conv1']

        return feature_layers

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

    def _content_loss(self, base: np.ndarray,
                      combination: np.ndarray) -> np.ndarray:
        """
        Loss function for content similiarly between base image and combination
        image

        By minimizing the loss, our generated combinated image will be
        consistent with the base image.
        """
        return K.sum(K.square(combination - base))

    def _gram_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Gram matrix of an image tensor
        (feature-wise outer product)
        """
        assert K.ndim(x) == 3
        if K.image_data_format == 'channels_first':
            features = K.batch_flatten(x)
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

        gram = K.dot(features, K.transpose(features))
        return gram

    def _style_loss(self, style: np.ndarray, combination: np.ndarray):
        """
        Loss function for style similiarity between style reference image and
        combination image. Based on gram matrices(which capture style) of
        feature maps from the style reference image and from the generated
        image
        """
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3
        S = self._gram_matrix(style)
        C = self._gram_matrix(combination)
        channels = 3
        size = self.img_nrows * self.img_ncols
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def _total_variation_loss(self, x: np.ndarray) -> np.ndarray:
        """
        Total loss is a combination of the other two loss functions,
        designed to keep the generated combined image locally
        coherent
        """
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(
                x[:, :, :self.img_nrows - 1, :self.img_ncols - 1] -
                x[:, :, 1:, :self.img_ncols - 1])
            b = K.square(
                x[:, :, :self.img_nrows - 1, :self.img_ncols - 1] -
                x[:, :, :self.img_nrows - 1, 1:])
        else:
            a = K.square(
                x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] -
                x[:, 1:, :self.img_ncols - 1, :])
            b = K.square(
                x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] -
                x[:, :self.img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def deconv(self, noise_ratio: float=0.025, content_weight: float=0.025,
               style_weight: float=1.0, total_variation_weight: float=1.0,
               output_layer: str='block5_conv1', iterations: int=15,
               output_steps: int=5, save_images: bool=True,
               save_dir: str='deconv_results',
               testing: bool=False) -> np.ndarray:
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
        base_image = K.variable(self.img_a)
        style_reference_image = K.variable(self.img_b)
        combination_image = K.placeholder((1,
                                           self.img_nrows,
                                           self.img_ncols,
                                           3))
        input_tensor = K.concatenate([base_image,
                                      style_reference_image,
                                      combination_image], axis=0)

        # Get Model
        model = self.model(include_top=False, weights='imagenet',
                           input_tensor=input_tensor)

        feature_maps = dict([(layer.name, layer.output)
                             for layer in model.layers])

        if output_layer not in feature_maps:
            raise IOError("input layer cannot be found in model")

        # feature_layers = self.features_list[:self.features_list.index(
            # output_layer)]

        # Setup Loss
        loss = K.variable(0.)

        # Calculate content loss
        layer_features = feature_maps[output_layer]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * self._content_loss(base_image_features,
                                                    combination_features)

        # Calculate style loss
        for layer in self.features_list:
            layer_features = feature_maps[layer]
            style_image_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self._style_loss(style_image_features, combination_features)
            loss += (style_weight / len(self.features_list)) * sl

        loss += total_variation_weight * self._total_variation_loss(
            combination_image)

        # Get the gradients of the combination image wrt the loss
        grads = K.gradients(loss, combination_image)

        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)

        f_outputs = K.function([combination_image], outputs)

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
        # combination image so as to minimize the content loss
        if testing:
            import tqdm
            range_fn = tqdm.trange
        else:
            range_fn = range

        for i in range_fn(iterations):
            input_img, _, _ = fmin_l_bfgs_b(evaluator.loss,
                                            input_img.flatten(),
                                            fprime=evaluator.grads,
                                            maxfun=20)
            if i % output_steps == 0:
                img = self._deprocess_image(input_img.copy())
                # save current combination image
                if save_images:
                    fname = '{}/from_{}_at_{}.png'.format(save_dir,
                                                          output_layer,
                                                          i)
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
