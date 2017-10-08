"""Patchmatch Implementation."""

import numpy as np
import os
import utils

from typing import Tuple


class Patchmatch(object):
    """Compute an NNF that maps patches in A/A' to B/B' or vice versa.

    """

    def __init__(self, A, Ap, B, Bp,
                 patchsize: int=7 // 2,
                 w: int=None,
                 alpha: float=0.5,
                 iterations: int=5) -> None:
        """Instantiate a PatchMatcher object.

        Parameters
        ----------
        A: ndarray
            The original structural image.
        Ap: ndarray
            The analogous, synthetic structural image.
        B: ndarray
            The analogous, synthetic style image.
        Bp: ndarray
            The original style image.
        NNF: ndarray or None
            An array containing the offsets. Either the upsampled NNF from the
            previous layer should be passed in or None. If None, an array with
            random offsets will be created.
        W: int
            Maximum search radius. If not specified then the maximum image
            dimension from image A will be used
        """
        # TODO(Use better getters and setters with validation checks)
        # TODO(add validation check for patchsize-> must be odds)
        if any(img is None for img in (A, Ap, B, Bp)):
            raise ValueError

        if A.shape != B.shape or Ap.shape != Bp.shape:
            raise ValueError

        self.A = A
        self.Ap = Ap
        self.B = B
        self.Bp = Bp
        self.patchsize = patchsize
        self.w = w or np.max(self.A.shape[0:2])
        self.alpha = alpha
        self.iterations = iterations
        self._curr_iteration = 0

        self.NNF_forward = self._random_init(self.A)
        self.NNF_reverse = self._random_init(self.B)
        self.source_patches_forward = utils.make_patch_matrix(
            self.A, self.patchsize)
        self.target_patches_forward = utils.make_patch_matrix(
            self.A, self.patchsize)
        self.source_patches_reverse = utils.make_patch_matrix(
            self.B, self.patchsize)
        self.target_patches_reverse = utils.make_patch_matrix(
            self.B, self.patchsize)
        self.NNF_D_forward = np.empty(
            (self.A.shape[0], self.A.shape[1])) * np.nan
        self.NNF_D_reverse = np.empty(
            (self.B.shape[0], self.B.shape[1])) * np.nan

    def _random_init(self,
                     source_image: np.ndarray) -> np.ndarray:
        """Initialize an NNF filled with random offsets

        :return ndarray
            Create NNF correspondence in three channels:
                (X-coord, Y-coord, Offsets)
            Note: offsets are represent a 2D displacement
        """

        im_shape = source_image.shape

        # First we need to create the displacement/offset matrices
        # Generate matrices of random x/y coordinates
        x = np.random.randint(low=0, high=im_shape[1],
                              size=(im_shape[0], im_shape[1]))
        y = np.random.randint(low=0, high=im_shape[0],
                              size=(im_shape[0], im_shape[1]))

        # Stack them
        f = np.dstack((y, x))

        # Now we generate a matrix g of size (im_shape[0], im_shape[1], 2)
        # such that g(y,x) = [y, x]
        g = utils.make_coordinates_matrix(im_shape)

        # Generate a NNF to be the differences of the two matrices
        f = f - g

        return f

    def _propagate_and_random_search(self,
                                     source_patches: np.ndarray,
                                     target_patches: np.ndarray,
                                     NNF: np.ndarray,
                                     NNF_D: np.ndarray) -> Tuple[np.ndarray,
                                                                 np.ndarray]:
        """Propagate and search adjacent good offsets

        Step 1: propagate
        Attempt to improve our NNF(x, y) using known offsets of NNF(x-1, y) and
        NNF(x, y-1). We aren't find good correspondences for every pixel of the
        images instead we look for best correspondence of some seeds(patche
        centers). Flow values are propagated from neighbor seeds to current
        seed if they have already been examined in the iteration.

        Note: Offsets are examined in scan order(from left to right, top to
        bottom) on odd iterations and reverse scan order on even iterations.
        """
        odd_iteration = self._curr_iteration % 2 != 0
        x_size, y_size = source_patches.shape[0], source_patches.shape[1]
        F = NNF.copy()
        D = NNF_D
        offset = 1

        # TODO(Why start out or end 2 pixels short?)
        if odd_iteration:
            start_x, start_y = 1, 1
            end_x = source_patches.shape[0] - 2
            end_y = source_patches.shape[1] - 2
            loop = 1
        else:
            start_x = source_patches.shape[0] - 2
            start_y = source_patches.shape[1] - 2
            end_x, end_y = 0, 0
            loop = -1

        k = int(np.ceil(-np.log10(self.w) / np.log10(self.alpha)))
        same_count = 0

        for i in range(start_x, end_x, loop):
            for j in range(start_y, end_y, loop):
                #
                # PROPOGATE
                #

                # TODO(PROPOGATE)
                # get the offsets f(x-1, y), and f(x, y-1)
                # compute distance D between:
                # A[x, y], B[f(x, y)], B[f(x-1, y) + (1, 0)]
                #   and B[f(x, y-1) + (0, 1)]
                # set f[x, y] to be argmin of computed distances
                # for even iterations iterate in reverse scan order
                # and examine patches

                # Get current distance from original pixel
                D_i, D_j = i + NNF[i, j, 0], j + NNF[i, j, 1]

                # Check if out of range [i, j] + v
                # If so limit to image dimensions
                D_i = np.clip(D_i, -x_size, x_size - 1)
                D_j = np.clip(D_j, -y_size, y_size - 1)

                _D_horizontal = None
                _D_verticle = None
                _D_orig = utils.compute_distance(source_patches[i, j],
                                                 target_patches[D_i, D_j])

                # Get offsets f(x-1, y) and f(x, y-1)
                v_horizontal = NNF[i + offset, j]
                v_verticle = NNF[i, j + offset]

                # Check if offsets are within bounds
                # If so calculate distance to offset
                if (v_horizontal[0] +
                    i < x_size and v_horizontal[1] +
                        j < y_size):
                    x_target = i + v_horizontal[0]
                    y_target = j + v_horizontal[1]

                    _D_horizontal = utils.compute_distance(
                        source_patches[i, j],
                        target_patches[x_target, y_target])
                else:
                    _D_horizontal = np.nan

                if (v_verticle[0] + i < x_size and v_verticle[1] + j < y_size):
                    x_target = i + v_verticle[0]
                    y_target = j + v_verticle[1]

                    _D_verticle = utils.compute_distance(
                        source_patches[i, j],
                        target_patches[x_target, y_target])
                else:
                    _D_verticle = np.nan

                # Possible distances with offsets
                _D = [_D_horizontal, _D_verticle, _D_orig]
                _NNF = [v_horizontal, v_verticle, NNF[i, j]]

                # Get min distance value
                min_val = np.nanmin(_D)

                # Check if all offsets are out of bounds
                if(np.isnan(min_val)):
                    continue

                # Get argmin distance
                index = _D.index(min_val)

                # SAME
                if D[i, j] == min_val:
                    same_count += 1

                # UPDATE
                elif np.isnan(D[i, j]) or min_val < D[i, j]:
                    D[i, j] = min_val
                    NNF[i, j] = _NNF[index]
                    F[i, j] = _NNF[index]

                #
                # RANDOM SEARCH
                #

                # TODO(RANDOM SEARCH)
                # u = v0 + w * a**i * R
                # i = 0
                # v0: current nearest neighbor f(x,y)
                # w: search radius. default max image dim.
                # alpha: scaling factor. default 1/2
                # i: 0, 1, 2, ... until w*a**i < 1
                # R: uniform random number in [-1, 1], [-1, 1]
                for l in range(k):
                    R = [np.random.uniform(-1, 1),
                         np.random.uniform(-1, 1)]

                    u = NNF[i, j] + np.multiply(
                        (self.alpha ** l) * self.w, R)

                    x = i + u[0]
                    y = j + u[1]

                    # Keep x and y coords in bounds
                    x = int(np.clip(x, -x_size, x_size - 1))
                    y = int(np.clip(y, -y_size, y_size - 1))

                    _D_r = utils.compute_distance(source_patches[i, j],
                                                  target_patches[x, y])

                    if _D_r < _D_orig:
                        F[i, j] = u
                        _D_orig = _D_r
        return F, D

    def propagate_and_random_search(self,
                                    write_images: bool=False,
                                    img_directory: str="results"):
        """Propagate and random search

        """
        # TODO(Implement)
        # repeat 5 times
        # on odd iterations:
        # iterate over all coords x,y in NNF
        #   from (left to right, top to bottom)
        # propagate down and right
        # random search
        # on odd iterations:
        # iterate over all coords x,y in NNF
        #   from (right to left, bottom to top)
        # propagate down and right
        # random search

        while self._curr_iteration < self.iterations:
            # Forward
            NNF_forward, NNF_D_forward = self._propagate_and_random_search(
                self.source_patches_forward,
                self.target_patches_forward,
                self.NNF_forward,
                self.NNF_D_forward)

            if write_images:
                if not os.path.isdir(img_directory):
                    os.mkdir(img_directory)

                filename = "nnf_forward_%d.png" % (self._curr_iteration + 1)
                utils.reconstruct_source_from_target(
                    self.B,
                    NNF_forward,
                    filename=os.path.join(img_directory, filename))

            # TODO(Backward)

            self._curr_iteration += 1
