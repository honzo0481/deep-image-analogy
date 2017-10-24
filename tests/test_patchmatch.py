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
def A6():
    A6 = np.zeros((6, 6, 3))
    A6[:3, 3:, :] += 0.25
    A6[3:, :3, :] += 0.5
    A6[3:, 3:, :] += 1.0
    return A6


@pytest.fixture
def Bp6():
    Bp6 = np.zeros((6, 6, 3))
    Bp6[:3, :3, :] += 1.0
    Bp6[:3, 3:, :] += 0.5
    Bp6[3:, :3, :] += 0.25
    return Bp6


@pytest.mark.skip('Not implemented.')
def test_patchmatcher_takes_2_imgs():
    """Patchmatcher should accept 2 images."""


@pytest.mark.skip('Not implemented.')
def test_patchmatcher_takes_4_imgs():
    """Patchmatcher should accept 2 images."""


@pytest.mark.skip('Not implemented.')
def test_patchmatcher_raises_ValueError_when_passed_3_images():
    """Patchmatcher should raise a ValueError if passed 3 images."""


def test_random_init_nnf_type(A6, Bp6):
    """Patchmatcher should return a randomly initialized nnf of type ndarray."""
    pm = Patchmatcher(A6, Bp6)
    assert isinstance(pm.NNF, np.ndarray)


@pytest.mark.parametrize('patchsize, padwidth', [(3, 1), (5, 2)])
def test_padwidth(A6, Bp6, patchsize, padwidth):
    """Padwith should be half the patchsize rounded down."""
    pm = Patchmatcher(A6, Bp6, patchsize=patchsize)
    assert pm.padwidth == padwidth


def test_nnfwidth(A6, Bp6):
    """nnfwidth should be input shape minus left padding and right padding."""
    pm = Patchmatcher(A6, Bp6, patchsize=3)
    assert pm.nnfwidth == 4


def test_use_precomputed_nnf(A6, Bp6, nnf):
    """Patchmatcher should accept a precomputed NNF as a param."""
    pm = Patchmatcher(A6, Bp6, NNF=nnf)
    assert np.allclose(nnf, pm.NNF)


def test_random_init_nnf_size(A6, Bp6):
    """_random_init should return an nnf with size equal to nnfwidth squared."""
    pm = Patchmatcher(A6, Bp6)
    nnfwidth = pm.nnfwidth
    assert pm.NNF.size == nnfwidth**2


@pytest.mark.parametrize('i, o, p, q0, q1, q2', [
    (0, 2, (0, 0), (0, 2), (0, 0), (1, 2)),
    (1, 4, (0, 1), (1, 0), (0, 3), (1, 2)),
    (2, 2, (0, 2), (0, 2), (1, 1), (1, 2)),
    (3, 11, (0, 3), (2, 3), (0, 3), (0, 3)),
    (4, 4, (1, 0), (1, 0), (3, 0), (1, 2)),
    (5, 5, (1, 1), (1, 1), (1, 1), (2, 0)),
    (6, 15, (1, 2), (3, 3), (1, 2), (1, 2)),
    (7, 5, (1, 3), (1, 1), (0, 0), (3, 3)),
    (8, 0, (2, 0), (0, 0), (1, 2), (2, 0)),
    (9, 3, (2, 1), (0, 3), (0, 1), (2, 1)),
    (10, 8, (2, 2), (2, 0), (1, 0), (0, 3)),
    (11, 7, (2, 3), (1, 3), (2, 1), (2, 1)),
    (12, 2, (3, 0), (0, 2), (2, 0), (1, 0)),
    (13, 2, (3, 1), (0, 2), (0, 3), (1, 3)),
    (14, 2, (3, 2), (0, 2), (0, 3), (3, 0)),
    (15, 15, (3, 3), (3, 3), (0, 3), (2, 3))
])
def test_get_patche_scan_order(i, o, p, q0, q1, q2, A6, Bp6, nnf):
    """In scan order _get_patch_centers returns offset plus neighbors below and left."""
    pm = Patchmatcher(A6, Bp6, NNF=nnf)
    assert pm._get_patch_centers(i, o) == (p, q0, q1, q2)


@pytest.mark.parametrize('i, o, p, q0, q1, q2', [
    (15, 15, (3, 3), (3, 3), (0, 1), (1, 3)),
    (14, 2, (3, 2), (0, 2), (3, 2), (3, 2)),
    (13, 2, (3, 1), (0, 2), (0, 1), (0, 0)),
    (12, 2, (3, 0), (0, 2), (0, 1), (3, 2)),
    (11, 7, (2, 3), (1, 3), (0, 1), (2, 3)),
    (10, 8, (2, 2), (2, 0), (1, 2), (3, 2)),
    (9, 3, (2, 1), (0, 3), (1, 3), (3, 2)),
    (8, 0, (2, 0), (0, 0), (0, 2), (3, 2)),
    (7, 5, (1, 3), (1, 1), (3, 3), (0, 3)),
    (6, 15, (1, 2), (3, 3), (1, 0), (1, 0)),
    (5, 5, (1, 1), (1, 1), (3, 2), (3, 3)),
    (4, 4, (1, 0), (1, 0), (1, 0), (3, 0)),
    (3, 11, (0, 3), (2, 3), (0, 3), (0, 1)),
    (2, 2, (0, 2), (0, 2), (2, 2), (2, 3)),
    (1, 4, (0, 1), (1, 0), (0, 1), (0, 1)),
    (0, 2, (0, 0), (0, 2), (0, 3), (0, 0))
])
def test_get_patch_centers_reverse_scan_order(i, o, p, q0, q1, q2, A6, Bp6, nnf):
    """In reverse scan order _get_patch_centers returns offset plus neighbors above and right."""
    pm = Patchmatcher(A6, Bp6, NNF=nnf)
    assert pm._get_patch_centers(i, o, scan_order=False) == (p, q0, q1, q2)


@pytest.mark.skip('Not implemented.')
def test_unidirectional_distance():
    """"""


@pytest.mark.skip('Not implemented.')
def test_bidirectional_distance():
    """"""


@pytest.mark.skip('Not implemented.')
def test_propagate_in_reverse_scan_order():
    """_propagate should return patches in scan order on odd iterations."""


@pytest.mark.skip('Not implemented.')
def test_random_search_gets_patches_at_exponentially_decreasing_dist():
    """"""


@pytest.mark.skip('Not implemented.')
def test_predict():
    """"""


@pytest.mark.skip('Not implemented.')
def test_get_patch():
    """"""
