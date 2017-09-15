import numpy as np

from collections import OrderedDict
from typing import List, Tuple, Dict


# TODO: Implement patch match algorithm
def patch_match(a: np.ndarray, b: np.ndarray, a_p: np.ndarray, b_p:
                np.ndarray, patch_size: Tuple, iterations: int,
                radius_max: int):
    """
    Implementation of the patchmatch algorithm for images

    Assume images have same size
    """
    raise NotImplementedError


# TODO: Build test cases
def generate_seed_indices(kernel_size: Tuple,
                          image_size: Tuple,
                          step: int) -> Dict[Tuple[int, int], List[int]]:
    w, h = image_size
    gridw, gridh = int(round(w / step)), int(round(h / step))
    xoffset = (w - (gridw - 1) * step) / 2
    yoffset = (h - (gridh - 1) * step) / 2
    numv = gridw * gridh

    # print("gW: {}, gH: {}".format(gridw, gridh))
    # print("X offset: {}".format(xoffset))
    # print("Y offset: {}".format(yoffset))
    # print("Num V: {}\n".format(numv))

    nbOffset = [(0, -1), (0, 1), (1, 0), (-1, 0), (-1, -1),
                (-1, 1), (1, -1), (1, 1)]

    seeds: Dict = OrderedDict()

    for i in range(numv):
        gridX = int(round(i % gridw))
        gridY = int(round(i / gridw))
        xy = (int(gridX * step + xoffset),
              int(gridY * step + yoffset))
        neighbors = []

        for j in range(8):
            nbGridX = gridX + nbOffset[j][0]
            nbGridY = gridY + nbOffset[j][1]
            if nbGridX < 0 or nbGridX >= gridw or nbGridY < 0 or nbGridY >= gridh:
                continue
            # TODO: Need to fix the neighbor approximation
            # neighbors.append(nbGridY * gridw + nbGridX)
            neighbors.append((nbGridX, nbGridY))

        seeds[xy] = neighbors

    return seeds
