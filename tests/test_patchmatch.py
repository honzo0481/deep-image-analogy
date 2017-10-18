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
    assert pmuni.NNF.size == nnflen**2


def test_propagate():
    """_propagate."""
    assert 0


@pytest.mark.parametrize('i, o, p', [
    (0, 2, (0, 0)),
    (1, 4, (0, 1)),
    (2, 2, (0, 2)),
    (3, 11, (0, 3)),
    (4, 4, (1, 0)),
    (5, 5, (1, 1)),
    (6, 15, (1, 2)),
    (7, 5, (1, 3)),
    (8, 0, (2, 0)),
    (9, 3, (2, 1)),
    (10, 8, (2, 2)),
    (11, 7, (2, 3)),
    (12, 2, (3, 0)),
    (13, 2, (3, 1)),
    (14, 2, (3, 2)),
    (15, 15, (3, 3))
])
def test_get_patches_gets_p(i, o, p, pmuni):
    """_get_patches should return the correct subset of offsets from the nnf."""
    print('nnflen: %s' % pmuni.nnflen)
    assert pmuni._get_patches(i, o)[0] == p


@pytest.mark.parametrize('i, o, q0', [
    (0, 2, (0, 2)),
    (1, 4, (1, 0)),
    (2, 2, (0, 2)),
    (3, 11, (2, 3)),
    (4, 4, (1, 0)),
    (5, 5, (1, 1)),
    (6, 15, (3, 3)),
    (7, 5, (1, 1)),
    (8, 0, (0, 0)),
    (9, 3, (0, 3)),
    (10, 8, (2, 0)),
    (11, 7, (1, 3)),
    (12, 2, (0, 2)),
    (13, 2, (0, 2)),
    (14, 2, (0, 2)),
    (15, 15, (3, 3))
])
def test_get_patches_gets_q0(i, o, q0, pmuni):
    """_get_patches should return the correct subset of offsets from the nnf."""
    assert pmuni._get_patches(i, o)[1] == q0


@pytest.mark.parametrize('i, o, q1', [
    (0, 2, (0, 0)),
    (1, 4, (0, 3)),
    (2, 2, (1, 1)),
    (3, 11, (0, 3)),
    (4, 4, (3, 0)),
    (5, 5, (1, 1)),
    (6, 15, (1, 2)),
    (7, 5, (0, 0)),
    (8, 0, (1, 2)),
    (9, 3, (0, 1)),
    (10, 8, (1, 0)),
    (11, 7, (2, 1)),
    (12, 2, (2, 0)),
    (13, 2, (0, 3)),
    (14, 2, (0, 3)),
    (15, 15, (0, 3))
])
def test_get_patches_gets_q1(i, o, q1, pmuni):
    """_get_patches should return the correct subset of offsets from the nnf."""
    assert pmuni._get_patches(i, o)[2] == q1


@pytest.mark.parametrize('i, o, q2', [
    (0, 2, (1, 2)),
    (1, 4, (1, 2)),
    (2, 2, (1, 2)),
    (3, 11, (0, 3)),
    (4, 4, (1, 2)),
    (5, 5, (2, 0)),
    (6, 15, (1, 2)),
    (7, 5, (3, 3)),
    (8, 0, (2, 0)),
    (9, 3, (2, 1)),
    (10, 8, (0, 3)),
    (11, 7, (2, 1)),
    (12, 2, (1, 0)),
    (13, 2, (1, 3)),
    (14, 2, (3, 0)),
    (15, 15, (2, 3))
])
def test_get_patches_gets_q2(i, o, q2, pmuni):
    """_get_patches should return the correct subset of offsets from the nnf."""
    assert pmuni._get_patches(i, o)[3] == q2
