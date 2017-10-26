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
    A = np.zeros((6, 6, 3))
    A[:3, 3:, :] += 0.25
    A[3:, :3, :] += 0.5
    A[3:, 3:, :] += 1.0
    return A


@pytest.fixture
def Bp():
    Bp = np.zeros((6, 6, 3))
    Bp[:3, :3, :] += 1.0
    Bp[:3, 3:, :] += 0.5
    Bp[3:, :3, :] += 0.25
    return Bp


def test_patchmatcher_takes_2_imgs(A, Bp):
    """Patchmatcher should accept 2 images."""
    pm = Patchmatcher(A, Bp)
    assert True


def test_patchmatcher_takes_4_imgs(A, Bp):
    """Patchmatcher should accept 4 images."""
    pm = Patchmatcher(A, Bp, A.copy(), Bp.copy())
    assert True


def test_patchmatcher_raises_ValueError_when_passed_3_images(A, Bp):
    """Patchmatcher should raise an ValueError if 3 images are passed in."""
    with pytest.raises(ValueError):
        pm = Patchmatcher(A, Bp, A.copy())


def test_random_init_nnf_type(A, Bp):
    """Patchmatcher should return a randomly initialized nnf of type ndarray."""
    pm = Patchmatcher(A, Bp)
    assert isinstance(pm.NNF, np.ndarray)


@pytest.mark.parametrize('patchsize, padwidth', [(3, 1), (5, 2)])
def test_padwidth(A, Bp, patchsize, padwidth):
    """Padwith should be half the patchsize rounded down."""
    pm = Patchmatcher(A, Bp, patchsize=patchsize)
    assert pm.padwidth == padwidth


def test_nnfwidth(A, Bp):
    """nnfwidth should be input shape minus left padding and right padding."""
    pm = Patchmatcher(A, Bp, patchsize=3)
    assert pm.nnfwidth == 4


def test_use_precomputed_nnf(A, Bp, nnf):
    """Patchmatcher should accept a precomputed NNF as a param."""
    pm = Patchmatcher(A, Bp, NNF=nnf)
    assert np.allclose(nnf, pm.NNF)


def test_random_init_nnf_size(A, Bp):
    """_random_init should return an nnf with size equal to nnfwidth squared."""
    pm = Patchmatcher(A, Bp)
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
def test_get_patche_scan_order(i, o, p, q0, q1, q2, A, Bp, nnf):
    """In scan order _get_patch_centers returns offset plus neighbors below and left."""
    pm = Patchmatcher(A, Bp, NNF=nnf)
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
def test_get_patch_centers_reverse_scan_order(i, o, p, q0, q1, q2, A, Bp, nnf):
    """In reverse scan order _get_patch_centers returns offset plus neighbors above and right."""
    pm = Patchmatcher(A, Bp, NNF=nnf)
    assert pm._get_patch_centers(i, o, scan_order=False) == (p, q0, q1, q2)


@pytest.mark.parametrize('coord, patch', [
    ((0, 0), np.zeros((3, 3, 3))),
    ((3, 3), np.ones((3, 3, 3))),
    ((2, 2), np.array([[[0., 0., 0.],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25]],
                       [[0.5, 0.5, 0.5],
                        [1., 1.,  1.],
                        [1., 1., 1.]],
                       [[0.5, 0.5, 0.5],
                        [1., 1., 1.],
                        [1., 1., 1.]]]))
])
def test_get_patch(coord, patch, A, Bp):
    """Get patch should return a padwidth X padwidth slice centered around input coord."""
    pm = Patchmatcher(A, Bp)
    assert np.allclose(pm._get_patch(coord, A), patch)


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


