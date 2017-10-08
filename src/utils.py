import cv2 as cv
import numpy as np

from typing import Tuple


def create_NNF_image(f):
    """Create an RGB image to visualize the nearest-neighbour field.

    """
    # square the individual coordinates
    magnitude = np.square(f)
    # sum the coordinates to compute the magnitude
    magnitude = np.sqrt(np.sum(magnitude, axis=2))
    # compute the orientation of each vector
    orientation = np.arccos(f[:, :, 1] / magnitude) / np.pi * 180
    # rescale the orientation to create a hue channel
    hue = np.array(orientation, np.uint8)
    # rescale the magnitude to create a saturation channel
    magnitude = magnitude / np.max(magnitude) * 255
    saturation = np.array(magnitude, np.uint8)
    # create a constant brightness channel
    brightness = np.zeros(magnitude.shape, np.uint8) + 200
    # create the HSV image
    hsv = np.dstack((hue, saturation, brightness))
    # return an RGB image with the specified HSV values
    rgb_image = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    return rgb_image


def create_NNF_vectors_image(source, target, f, patch_size,
                             server=True,
                             subsampling=100,
                             line_width=0.5,
                             line_color='k',
                             tmpdir='./'):
    """Display the nearest-neighbour field

    as a sparse vector field between source and target images
    """
    import matplotlib.pyplot as plt

    # get the shape of the source image
    im_shape = source.shape

    # if you are using matplotlib on a server
    if server:
        plt.switch_backend('agg')
    import matplotlib.patches as patches

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    source = cv.cvtColor(source, cv.COLOR_BGR2RGB)
    target = cv.cvtColor(target, cv.COLOR_BGR2RGB)

    # create an image that contains the source and target side by side
    plot_im = np.concatenate((source, target), axis=1)
    ax.imshow(plot_im)

    vector_coords = make_coordinates_matrix(im_shape, step=subsampling)
    vshape = vector_coords.shape
    vector_coords = np.reshape(vector_coords, (vshape[0] * vshape[1], 2))

    for coord in vector_coords:
        rect = patches.Rectangle(
            (coord[1] - patch_size / 2.0,
             coord[0] - patch_size / 2.0),
            patch_size,
            patch_size,
            linewidth=line_width,
            edgecolor=line_color,
            facecolor='none')
        ax.add_patch(rect)

        arrow = patches.Arrow(coord[1],
                              coord[0],
                              f[coord[0], coord[1], 1] + im_shape[1],
                              f[coord[0], coord[1], 0],
                              lw=line_width,
                              edgecolor=line_color)
        ax.add_patch(arrow)

    dpi = fig.dpi
    fig.set_size_inches(im_shape[1] * 2 / dpi, im_shape[0] / dpi)
    tmp_image = tmpdir + '/tmpvecs.png' \

    fig.savefig(tmp_image)
    plt.close(fig)
    return tmp_image


def save_NNF(f, filename):
    """
    Save the nearest-neighbour field matrix in numpy file
    """
    try:
        np.save('{}'.format(filename), f)
    except IOError as e:
        return False, e
    else:
        return True, None


def load_NNF(filename, shape=None):
    """
    Load the nearest-neighbour field from a numpy file
    """
    try:
        f = np.load(filename)
    except IOError as e:
        return False, None, e
    else:
        if shape is not None:
            if (f.shape[0] != shape[0] or
                    f.shape[1] != shape[1]):
                return False, None, 'NNF has incorrect dimensions'
        return True, f, None


def make_coordinates_matrix(im_shape: Tuple,
                            step: int=1) -> np.ndarray:
    """Return a coordinate matrix

    :return np.ndarray
        size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))


def make_patch_matrix(im: np.ndarray,
                      patch_size: int) -> np.ndarray:
    """Create a patch matrix

    :return ndarray
        shape (X, Y, C, P^2)
    """
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = (im.shape[0] + patch_size - 1,
                    im.shape[1] + patch_size - 1,
                    im.shape[2])
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel.
    # If the original image had NxM pixels, this matrix will have
    # NxMx(patch_size*patch_size) pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = \
                padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


def compute_distance(source_patch: np.ndarray,
                     target_patch: np.ndarray) -> int:
    """Calculate sum of squared differences between two patches

    """
    v1 = np.ndarray.flatten(source_patch)
    v2 = np.ndarray.flatten(target_patch)

    dist = np.abs(v1 - v2)

    return dist.dot(dist)


def compute_distance_2(source_patch: np.ndarray,
                       target_patch: np.ndarray) -> int:
    """Calculate sum of squared differences between two patches

    """
    tmp = source_patch[:, None, :] - target_patch
    return np.sum(np.einsum('ijk,ijk->ij', tmp, tmp))


def compute_distance_3(source_patch: np.ndarray,
                       target_patch: np.ndarray) -> int:
    """Calculate sum of squared differences between two patches

    """
    return np.sum((source_patch - target_patch)**2)


def reconstruct_source_from_target(target: np.ndarray,
                                   NNF: np.ndarray,
                                   write_image: bool=True,
                                   filename: str=None) -> np.ndarray:
    """Used a computed NNF to reconstruct the source image

    """
    rec_source = np.zeros(target.shape)

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):

            x = (i + NNF[i, j, 0])
            y = (j + NNF[i, j, 1])

            rec_source[i, j] = target[x, y]

    if write_image:
        # with open(filename, 'w') as f:
        cv.imwrite(filename, rec_source)

    return rec_source
