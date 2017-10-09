"""Patchmatch tests."""
import cv2 as cv
import numpy as np
import os
import pytest

from patchmatch import Patchmatch
from utils import compute_distance


def test_compute_distance():
    a = np.asarray([[1, 2, 3],
                    [4, 5, 6],
                    [6, 7, 8]])
    b = a + 1

    assert compute_distance(a, b) == 9


class Data(object):
    def __init__(self):
        self.images = {
            'A': None,
            'B': None,
        }


@pytest.fixture(scope="module")
def data():
    return Data()


@pytest.mark.incremental
class TestPatchmatch(object):
    def _test_image(self, image):
        assert isinstance(image, str)
        path = os.path.abspath(image)
        assert os.path.exists(path), "File doesn't exist"
        assert os.path.isfile(path), "Path isn't a file"
        return path

    def test_image_files(self, data, style_image, content_image):
        style_path = self._test_image(style_image)
        content_path = self._test_image(content_image)
        data.images['A'] = style_path
        data.images['B'] = content_path

    def test_image_shape(self, data):
        img_A = cv.imread(data.images['A'])
        img_B = cv.imread(data.images['B'])
        assert img_A.shape == img_B.shape,\
            "Images must be the same shape"
        data.images['A'] = img_A
        data.images['B'] = img_B

    def test_patchmatch_init(self, data):
        A, B = data.images['A'], data.images['B']

        patchsize = 7 // 2

        with pytest.raises(ValueError, message="Image data cannot be None"):
            Patchmatch(A, None)

        with pytest.raises(ValueError,
                           messages="Images should have similiar shapes"):
            Patchmatch(np.zeros((10, 9, 3)),
                       np.zeros((20, 5, 3)))

        pm = Patchmatch(A, B, patchsize=patchsize)
        assert pm.NNF.shape[:-1] == A.shape[:-1],\
            "Nearest-neighbor field(NNF) should be the same size as image A"
        assert pm.NNF.shape[:-1] == B.shape[:-1],\
            "Nearest-neighbor field(NNF) should be the same size as image B"
        assert pm.NNF.shape[-1] == 2,\
            "Make sure we have a 2D displacement"
        assert pm.NNF.shape[-1] == 2,\
            "Make sure we have a 2D displacement"

        assert pm.source_patches.shape[:-1] == A.shape
        assert pm.source_patches.shape[-1] == patchsize ** 2
        assert pm.NNF_D.shape == A.shape[:-1]

        pm2 = Patchmatch(A, B)

        assert not np.array_equal(pm.NNF, pm2.NNF),\
            "Not randomly initialized"

    def test_propagate_and_random_search(self, data):
        A, B = data.images['A'], data.images['B']
        pm = Patchmatch(A, B,
                        patchsize=7 // 2,
                        iterations=5)
        pm.propagate_and_random_search(write_images=True,
                                       verbose=True)
