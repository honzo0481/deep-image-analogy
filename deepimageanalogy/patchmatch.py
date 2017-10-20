"""Patchmatch Implementation."""

import numpy as np


def  bidirectional_distance(p, q, A, Bp, B, Ap):
    """Return the bidirectionally weighted distance D between patches P and Q.

    Where p and q are patch centers in the source and target images respectively.

    Parameters
    ----------
    p: 

    q: 

    A: ndarray
        The original structural image.
    Ap: ndarray
        The analogous, synthetic structural image.
    B: ndarray
        The analogous, synthetic style image.
    Bp: ndarray
        The original style image.

    Returns
    ------
    d: float32
        the computed distance.
    """


def unidirectional_distance(p, q, A, Bp):
    """Return the unidirectional distance D between patches P and Q.

    Where p and q are patch centers in the source and target images respectively.

    Parameters
    ----------
    p: 

    q: 

    A: ndarray
        The original structural image.
    Bp: ndarray
        The original style image.

    Returns
    ------
    d: float32
        the computed distance.
    """


class Patchmatcher(object):
    """Compute an NNF that maps patches from A/A' -> B/B' or vice versa."""

    def __init__(self,
                 A,
                 Bp,
                 B=None,
                 Ap=None,
                 NNF=None,
                 dist_metric=None,
                 patchsize=3, 
                 w=None, 
                 alpha=0.5):
        """Instantiate a PatchMatcher object.

        Parameters
        ----------
        A:

        Bp:

        B:

        Ap:

        NNF: ndarray or None
            An array containing the offsets. Either the upsampled NNF from the
            previous layer should be passed in or None. If None, an array with
            random offsets will be created.

        dist_metric: 

        patchsize:

        w:

        alpha:

        """
        if (B is not None and Ap is None) or (B is None and Ap is not None):
            raise ValueError('B and Ap must be provided together. One is missing.')

        self.A = A
        self.Bp = Bp

        if B is not None and Ap is not None:
            self.B = B
            self.Ap = Ap

        # TODO: handle case when bidirectional_distance is provided with only 
        # two images instead of four. 
        if dist_metric:
            self.dist_metric = dist_metric
        else:
            self.dist_metric = bidirectional_distance
        
        self.patchsize = patchsize
        # Floor division gives the before and after pad widths
        self.padwidth = self.patchsize // 2

        self.nnflen = self.A.shape[0] - self.padwidth * 2

        self.w = w or self.nnflen
        self.alpha = alpha
        # for now assume the first two input dims will always be square.
        if NNF is None:
            self.NNF = self._random_init()
        else:
            self.NNF = NNF

    def _random_init(self):
        """Initialize an NNF filled with random offsets."""
        NNF = np.random.randint(self.nnflen**2, size=self.nnflen**2, dtype='int')

        return NNF

    def _get_patches(self, index, offset, forward=True):
        """"""
        s = self.nnflen
        # the patch center in the source image. z in the Barnes paper.
        p = index // s, index % s
        # candidate patch center in the target image. f(z)
        q0 = offset // s, offset % s
        # another candidate patch center. f(z - [1,0]) + [1,0]
        q1 = (self.NNF[index-1]+1) % len(self.NNF) // s, (self.NNF[index-1]+1) % len(self.NNF) % s
        # last candidate. f(z - [0,1]) + [0,1]
        q2 = (self.NNF[index-s]+s) % len(self.NNF) // s, (self.NNF[index-s]+s) % len(self.NNF) % s

        return p, q0, q1, q2

    def _propagate(self, even):
        """Propagate adjacent good offsets."""
        # get the offsets f(x-1, y), and f(x, y-1)
        # TODO: still need to clamp the q values that fall outside the nnf bounds.
        for index, offset in enumerate(self.NNF):
            p, q0, q1, q2 = self._get_patches(index, offset)

            # compute distance D between:
            # A[x, y], B[f(x, y)], B[f(x-1, y) + (1, 0)], and B[f(x, y-1) + (0, 1)]
            d0 = self.dist_metric(p, q0)
            d1 = self.dist_metric(p, q1)
            d2 = self.dist_metric(p, q2)
            # set f[x, y] to be argmin of computed distances
            if d0 > d1:
                self.NNF[index] = q1[0]*self.nnflen + q1[1]
                d0 = d1
            if d0 > d2:
                self.NNF[index] = q2[0]*self.nnflen + q2[1]
            # TODO: for even iterations iterate in reverse scan order and examine patches

    def _random_search(self):
        """Search for good offsets at exponentially descreasing distances."""
        # u = v0 + w * a**i * R
        i = 0
        # v0: current nearest neighbor f(x,y)
        # w: search radius. default max image dim.
        # alpha: scaling factor. default 1/2
        # i: 0, 1, 2, ... until w*a**i < 1
        # R: uniform random number in [-1, 1], [-1, 1]

    def predict(self):
        """Return an nnf."""
        # repeat 5 times
        # on odd iterations:
            # iterate over all coords x,y in NNF from (left to right, top to bottom)
            # propagate down and right
            # random search
        # on odd iterations:
            # iterate over all coords x,y in NNF from (right to left, bottom to top)
            # propagate down and right
            # random search
        for i in range(1, 6):
            even = i % 2
            self._propagate(even=even)
            self._random_search()
