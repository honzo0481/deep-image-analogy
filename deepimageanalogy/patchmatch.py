"""Patchmatch Implementation."""

import numpy as np


def  bidirectional_distance(p, q, A, Ap, B, Bp):
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


def unidirectional_distance(p, q, A, B):
    """Return the unidirectional distance D between patches P and Q.

    Where p and q are patch centers in the source and target images respectively.

    Parameters
    ----------
    p: 

    q: 

    A: ndarray
        The original structural image.
    B: ndarray
        The analogous, synthetic style image.

    Returns
    ------
    d: float32
        the computed distance.
    """


class Patchmatcher(object):
    """Compute an NNF that maps patches from A/A' -> B/B' or vice versa."""

    def __init__(self, dist_metric, NNFsize=None, NNF=None, patchsize=3, w=None, alpha=0.5):
        """Instantiate a PatchMatcher object.

        Parameters
        ----------
        dist_metric: 

        NNFsize:

        NNF: ndarray or None
            An array containing the offsets. Either the upsampled NNF from the
            previous layer should be passed in or None. If None, an array with
            random offsets will be created.

        patchsize:

        w:

        alpha:
        
        """
        if not NNFsize and not NNF:
            raise ValueError('Either NNF or NNFsize must be provided.')

        if NNF:
            self.NNFsize = NNF.shape[0]
        else:
            self.NNFsize = NNFsize

        self.dist_metric = dist_metric
        self.patchsize = patchsize
        # Floor division gives the before and after pad widths
        self.padwidth = self.patchsize // 2
        self.w = w or self.NNFsize
        self.alpha = alpha
        # for now assume the first two input dims will always be square.
        self.NNF = NNF or self._random_init()

    def _random_init(self):
        """Initialize an NNF filled with random offsets."""
        NNF = np.random.randint(self.NNFsize, size=(self.NNFsize**2), dtype='int')

        return NNF

    def _propagate(self, even):
        """Propagate adjacent good offsets."""
        # get the offsets f(x-1, y), and f(x, y-1)
        s = self.NNFsize
        for index, offset in enumerate(self.NNF):
            # the patch center in the source image. z in the Barnes paper.
            p = index // s, index % s
            # candidate patch in the target image. f(z) 
            q0 = offset // s, offset % s
            # another candidate patch. f(z - [1,0]) + [1,0]
            q1 = (self.NNF[index-1]+1) // s, (self.NNF[index-1]+1) % s
            # last candidate. f(z - [0,1]) + [0,1]
            q2 = (self.NNF[index-s]+s) // s, (self.NNF[index-1]+1) % s

        # compute distance D between:
        # A[x, y], B[f(x, y)], B[f(x-1, y) + (1, 0)], and B[f(x, y-1) + (0, 1)]
        # set f[x, y] to be argmin of computed distances
        # for even iterations iterate in reverse scan order and examine patches

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