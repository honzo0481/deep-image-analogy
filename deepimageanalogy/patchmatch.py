"""Patchmatch Implementation."""

import numpy as np

def  bidirectional_distance(p, q):
    """Return the bidirectional distance D between patches P and Q.
    """



class Patchmatcher(object):
    """Compute an NNF that maps patches from A/A' -> B/B' or vice versa."""

    def __init__(self, dist_metric='bi', patchsize=3, w=None, alpha=0.5, NNF=None):
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
        """
        if dist_metric == 'bi':
            self.dist_metric = bidirectional_distance

        self.patchsize = patchsize
        # Floor division gives the before and after pad widths
        self.padwidth = self.patchsize // 2
        self.w = w or self.padwidth
        self.alpha = alpha
        self.NNF = NNF or self._random_init()

    def _random_init(self):
        """Initialize an NNF filled with random offsets."""
        # for now assume the first two input dims will always be square.
        length = self.A.shape[0] - self.padwidth*2
        # NOTE: in this implementation padding is not included during initialization.
        NNF = np.random.randint(length, size=(length**2), dtype='int')

        return NNF

    def _propagate(self, even):
        """Propagate adjacent good offsets."""
            # get the offsets f(x-1, y), and f(x, y-1)
            for index, offset in enumerate(self.NNF):
                p = index // l, i % l
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