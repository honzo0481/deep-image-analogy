"""Patchmatch tests."""

import pytest
import numpy as np
from deepimageanalogy.patchmatch import Patchmatcher


@pytest.fixture
def nnf():
    return np.array([
        # 0, 11, 15, 12, 9, 6, 4, 7, 7, 9, 14, 15, 12, 10, 0, 12
        2, 4, 2, 11, 4, 5, 15, 5, 0, 3, 8, 7, 2, 2, 2, 15
    ])


@pytest.fixture
def A():
    return np.random.rand(6, 6, 3)


@pytest.fixture
def Bp():
    return np.random.rand(6, 6, 3)


def test_random_init_nnf_type(A, Bp):
    """Patchmatcher should return a randomly initialized nnf of type ndarray."""
    pm = Patchmatcher(A, Bp)
    assert isinstance(pm.NNF, np.ndarray)


@pytest.mark.parametrize('patchsize, padwidth', [(3, 1), (5, 2)])
def test_padwidth(A, Bp, patchsize, padwidth):
    """Padwith should be half the patchsize rounded down."""
    pm = Patchmatcher(A, Bp, patchsize=patchsize)
    assert pm.padwidth == padwidth


def test_nnflen(A, Bp):
    """Nnflen should be input shape minus left padding and right padding."""
    pm = Patchmatcher(A, Bp, patchsize=3)
    assert pm.nnflen == 4


def test_use_precomputed_nnf(A, Bp, nnf):
    """Patchmatcher should accept a precomputed NNF as a param."""
    pm = Patchmatcher(A, Bp, NNF=nnf)

    assert np.allclose(nnf, pm.NNF)


def test_random_init_nnf_size(A, Bp):
    """_random_init should return an nnf with size equal to nnflen squared."""
    pm = Patchmatcher(A, Bp)
    nnflen = pm.nnflen
    assert pm.NNF.size == nnflen**2


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
def test_get_patches_gets_p(i, o, p, A, Bp, nnf):
    """_get_patches should return the correct subset of offsets from the nnf."""
    pm = Patchmatcher(A, Bp, NNF=nnf)
    assert pm._get_patches(i, o)[0] == p


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
def test_get_patches_gets_q0(i, o, q0, A, Bp, nnf):
    """_get_patches should return the correct subset of offsets from the nnf."""
    pm = Patchmatcher(A, Bp, NNF=nnf)
    assert pm._get_patches(i, o)[1] == q0


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
def test_get_patches_gets_q1(i, o, q1, A, Bp, nnf):
    """_get_patches should return the correct subset of offsets from the nnf."""
    pm = Patchmatcher(A, Bp, NNF=nnf)
    assert pm._get_patches(i, o)[2] == q1


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
def test_get_patches_gets_q2(i, o, q2, A, Bp, nnf):
    """_get_patches should return the correct subset of offsets from the nnf."""
    pm = Patchmatcher(A, Bp, NNF=nnf)
    assert pm._get_patches(i, o)[3] == q2

@pytest.mark.skip('Not implemented.')
def test_propagate():
    """_propagate."""
