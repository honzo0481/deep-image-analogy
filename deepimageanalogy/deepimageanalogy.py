"""Core module for the package."""

import numpy as np

from preprocessing import get_feature_pyramids
from patchmatch import patchmatch
import argparse


def deep_image_analogy(A, Bp):
    """Return two image analogies A' and B given A and B'."""
    # 4.1: preprocessing
    FA, FBp = get_feature_pyramids(A, Bp)
    FAp, FB = {}, {}
    # the coarsest layers in the synthetic feature pyramids are the same as the
    # extracted layers
    FAp[5], FB[5] = np.copy(FA[5]), np.copy(FBp[5])

    # initialize the chi5 NNFs with random offsets.

    # 4.2: Nearest-neighbor Field Search

    # 4.3: Latent Image Reconstruction

    # 4.4: Nearest-neighbor Field Upsampling

    # 4.5: Output


def main():
    """CLI for the package."""
    parser = argparse.ArgumentParser(description='Make image analogies.')
    parser.add_argument('A')
    parser.add_argument('Bp')
    args = parser.parse_args()

    deep_image_analogy(args.A, args.Bp)


if __name__ == '__main__':
    main()
