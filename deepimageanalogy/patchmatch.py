"""Patchmatch Implementation."""

import numpy as np


class Patchmatcher:
    """compute an NNF that maps patches in A/A' to B/B' or vice versa."""

    def __init__(self, A, Ap, B, Bp, NNF=None):
        """"instantiate a PatchMatcher object.

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
        self.A = A
        self.Ap = Ap
        self.B = B
        self.Bp = Bp
        self.NNF = NNF or self._random_init()

    def _random_init(self):
        """Initialize an NNF filled with random offsets."""
        # Start with an empty array of shape A.rows X A.cols X 2
        NNF = np.empty((*self.A.shape, 2), dtype=np.uint)
        return NNF

    def _propagate(self):
        """Propagate adjacent good offsets."""

    def _random_search(self):
        """Search for good offsets at exponentially descreasing distances."""

    def predict(self):
        """Return an nnf."""
