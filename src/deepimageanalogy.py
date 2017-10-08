"""Core module for the package."""

import argparse
import os

from preprocessing import get_feature_pyramids


class DeepAnalogy(object):
    """Compute Image Analogies between two image pairs

    1. Preprocessing
    2. NNF Search
    3. Latent Image Reconstruction
    4. NNF Upsampling
    5. Output
    """

    def __init__(self, path_A: str='', path_BP: str='',
                 output_dir: str='', blend_weight: int=3,
                 ratio: float=0.5, photo_transfer: bool=False) -> None:

        self.path_A = path_A
        self.path_BP = path_BP
        self.output_dir = output_dir
        self.blend_weight = blend_weight
        self.ratio = ratio
        self.photo_transfer = photo_transfer
        self.FA = None
        self.FBp = None

    @property
    def path_A(self):
        return self.__path_A

    @path_A.setter
    def path_A(self, path_A: str):
        if not isinstance(path_A, str):
            raise ValueError
        # check if path exists
        if path_A != '' and not os.path.isfile(path_A):
            raise IOError
        self.__path_A = path_A

    @property
    def path_BP(self):
        return self.__path_BP

    @path_BP.setter
    def path_BP(self, path_BP: str):
        if not isinstance(path_BP, str):
            raise ValueError
        # check if path exists
        if path_BP != '' and not os.path.isfile(path_BP):
            raise IOError
        self.__path_BP = path_BP

    @property
    def output_dir(self):
        return self.__output_dir

    @output_dir.setter
    def output_dir(self, output_dir: str):
        if not isinstance(output_dir, str):
            raise ValueError
        if output_dir != '' and not os.path.isdir(output_dir):
            raise IOError
        self.__output_dir = output_dir

    # @property
    # def blend_weight(self):
    #     return self.__blend_weight

    # @blend_weight.setter
    # def __blend_weight(self, blend_weight: int):
    #     if type(blend_weight) != int:
    #         raise ValueError
    #     self.__blend_weight = blend_weight

    # @property
    # def ratio(self):
    #     return self.__ratio

    # @ratio.setter
    # def __ratio(self, ratio: float):
    #     if type(ratio) != float:
    #         raise ValueError
    #     self.__ratio = ratio

    # @property
    # def photo_transfer(self):
    #     return self.__photo_transfer

    # @photo_transfer.setter
    # def __photo_transfer(self, photo_transfer: bool):
    #     if type(photo_transfer) != bool:
    #         raise ValueError
    #     self.__photo_transfer = photo_transfer

    def __str__(self):
        msg = []
        msg.append("Path to A: {}".format(self.path_A))
        msg.append("Path to B': {}".format(self.path_BP))
        msg.append("Output Dir: {}".format(self.output_dir))
        msg.append("Ratio: {}".format(self.ratio))
        msg.append("Blend Weight: {}".format(self.blend_weight))
        msg.append("Photo Transfer: {}".format(self.photo_transfer))
        return '\n'.join(msg)

    def load_inputs(self):
        self.FA, self.FBp = get_feature_pyramids(self.path_A, self.path_BP)

    def compute_ann(self):

        # Build Classifiers for input images A and B
        # Get feature maps

        # Feature Match; iterate through layers
        # Normalize
        # Patchmatch
        # Deconv
        # Upsample
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make image analogies.')
    parser.add_argument('A')
    parser.add_argument('Bp')
    args = parser.parse_args()

    dia = DeepAnalogy(args.A, args.Bp)
