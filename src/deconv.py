# -*- coding: utf-8 -*-
"""
Deconvolution Step

"""

import keras.backend as K
import numpy as np
import os

from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


class Deconv(object):

    def __init__(self, model, preprocess_input, path_A):
        self.model = model
        self.preprocess_input = preprocess_input
        if not os.path.isfile(path_A):
            raise IOError
        self.path_A = path_A
        self.img_ncols, self.img_nrows = load_img(path_A).size

    def _preprocess_image(self):
        img = load_img(self.path_A,
                       target_size=(self.img_ncols, self.img_nrows))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input(img)
        return img

    def _deprocess_image(self, x):
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

    def _content_loss(self, base, gen):
        return K.sum(K.square(gen - base))

    def deconv(self, noise_ratio=1.0, content_weight=1.0, layer='block5_conv1',
               iterations=15, output_steps=5, save_images=True,
               save_dir='deconv_results'):

        img_a = self._preprocess_image()

        a_image = K.variable(img_a)
        generated_image = K.placeholder((1, self.img_nrows, self.img_ncols, 3))

        input_tensor = K.concatenate([a_image,
                                      generated_image], axis=0)

        model = self.model(include_top=False, weights='imagenet',
                           input_tensor=input_tensor)

        feature_maps = dict([(layer.name, layer.output)
                             for layer in model.layers])

        # loss
        loss = K.variable(0.)
        layer_features = feature_maps[layer]
        a_features = layer_features[0, :, :, :]
        gen_features = layer_features[1, :, :, :]
        loss += content_weight * self._content_loss(a_features, gen_features)

        # get the gradients of the generated image wrt the loss
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
        x = noise_ratio * noise_img + (1. - noise_ratio) * img_a

        if save_images:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # run scipy-based optimization (L-BFGS) over the pixels of the
        # generated image so as to minimize the content loss
        for i in range(iterations):
            x, _, _ = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                    fprime=evaluator.grads, maxfun=20)
            # save current generated image
            if i % output_steps == 0:
                img = self._deprocess_image(x.copy())
                if save_images:
                    fname = '{}/from_{}_at_{}.png'.format(save_dir, layer, i)
                    imsave(fname, img)

        return self._deprocess_image(x.copy())


class Evaluator(object):

    def __init__(self, f_outputs):
        self.loss_value = None
        self.grads_values = None
        self._f_outputs = f_outputs

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self._eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def _eval_loss_and_grads(self, x, size=(448, 448)):
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
