"""Patchmatch tests."""

from deepimageanalogy.patchmatch import Patchmatcher


def test_random_init(A):
    """_init_nnf should return an nnf of the correct size with random offsets."""
    pm = Patchmatcher()
    NNF = pm._random_init()
    assert NNF.shape == A.shape
