"""Patchmatch Implementation."""

import numpy as np


class Patchmatcher:
    """compute an NNF that maps patches in A/A' to B/B' or vice versa."""

    def __init__(self, A, Ap, B, Bp, patchsize=3, NNF=None):
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
        self.A = A
        self.Ap = Ap
        self.B = B
        self.Bp = Bp
        self.NNF = NNF or self._random_init()

    def _random_init(self):
        """Initialize an NNF filled with random offsets."""
        # for now assume the first two input dims will always be square.
        # The subtraction accounts for padding.
        length = self.A.shape[0] - 2

        NNF = np.empty((length, length, 2), dtype='int')
        for i, ix in enumerate(np.ix_(np.arange(length), np.arange(length))):
            NNF[..., i] = ix

        offsets = np.random.randint(length, size=(length, length, 2), dtype='int')

        NNF = offsets - NNF

        NNF = np.pad(NNF, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)

        return NNF

    def _propagate(self):
        """Propagate adjacent good offsets."""

    def _random_search(self):
        """Search for good offsets at exponentially descreasing distances."""

    def predict(self):
        """Return an nnf."""
