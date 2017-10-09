"""Patchmatch tests."""

import pytest
import numpy as np


SEED = 1234567890

np.random.seed(SEED)


@pytest.fixture
def pmuni():
    """Patchmatcher object fixture."""
    from deepimageanalogy.patchmatch import Patchmatcher

    A = np.random.rand(6, 6, 3)
    Bp = np.random.rand(6, 6, 3)

    return Patchmatcher(A, Bp)


def test_random_init_nnf_size(pmuni):
    """_random_init should return an nnf with size equal to nnflen squared."""

    nnflen = pmuni.nnflen
    print(pmuni.NNF)
    assert pmuni.NNF.size == nnflen**2
    assert 0


def test_propagate_handles_corners_correctly():
    """_propagate indexes (q0), (q0, q1), (q0, q2), (q0, q1, q2) 

    iterating over corners in scan order.
    """


def test_propagate_handles_edges_correctly():
    """_propagate indexes (q0, q1), (q0, q2), (q0, q1, q2), (q0, q1, q2) 

    iterating over upper, left, right and lower edges in scan order.
    """

