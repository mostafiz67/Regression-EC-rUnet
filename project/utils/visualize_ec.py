
from pathlib import Path
from typing import Any, List, Tuple, Union
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import Figure
from numpy import ndarray
# from utils.const import PARAMETRIC_MAP


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    """Apply a 3D binary mask to a 1-channel, 3D ndarray `img` by creating a 3-channel
    image with masked regions shown in transparent blue."""
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)  # img
    # scaled = np.array(img * 255, dtype=int)
    return scaled

class BrainSlices:
    def __init__(
        self,
        ratio_ec_data: ndarray,
        ratio_diff_ec_data: ndarray,
        ratio_sign_ec_data: ndarray,
        ratio_diff_sign_ec_data: ndarray,
        inter_union_vox_ec_data: ndarray
    ):
        self.fig_data_1_img: ndarray = make_imgs(ratio_ec_data)
        self.fig_data_2_img: ndarray = make_imgs(ratio_diff_ec_data)
        self.fig_data_3_img: ndarray = make_imgs(ratio_sign_ec_data)
        self.fig_data_4_img: ndarray = make_imgs(ratio_diff_sign_ec_data)
        self.fig_data_5_img: ndarray = make_imgs(inter_union_vox_ec_data)

        si, sj, sk = 128, 128, 128
        i = si // 2
        j = sj // 2
        k = sk // 2

        self.slices = [ self.get_slice(self.fig_data_1_img, i, j, k), 
                        self.get_slice(self.fig_data_2_img, i, j, k),
                        self.get_slice(self.fig_data_3_img, i, j, k), 
                        self.get_slice(self.fig_data_4_img, i, j, k),
                        self.get_slice(self.fig_data_5_img, i, j, k), 
                        ]


        self.title = ["Ratio",
        "Ratio-diff",
        "Ratio-signed",
        "Ratio-diff-signed",
        "Intersection-Union"]

    def get_slice(self, input: ndarray, i: int, j: int, k: int) -> List[Tuple[ndarray, ...]]:
        return [
            (input[i // 2, ...], input[i, ...], input[i + i // 2, ...]),
            (input[:, j // 2, ...], input[:, j, ...], input[:, j + j // 2, ...]),
            (input[:, :, k // 2, ...], input[:, :, k, ...], input[:, :, k + k // 2, ...]),
        ]

    def plot(self) -> Figure:
        nrows, ncols = len(self.slices), 3  # one row for each slice position
        fig = plt.figure(figsize=(13, 10)) # fig = plt.figure(figsize=(13, 10))
        gs = gridspec.GridSpec(nrows, ncols)

        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 3])
            ax2 = plt.subplot(gs[i * 3 + 1])
            ax3 = plt.subplot(gs[i * 3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                if i == 0:
                    axis.set_title(self.title[0])
                elif i ==1:
                    axis.set_title(self.title[1])
                elif i ==2:
                    axis.set_title(self.title[2])
                elif i ==3:
                    axis.set_title(self.title[3])
                elif i ==4:
                    axis.set_title(self.title[4])
        plt.tight_layout(pad=3,h_pad=0.0, w_pad=0.1) # plt.tight_layout(pad=3, h_pad=0.0, w_pad=0.1)
        fig.suptitle('Parametric map images from EC metrics (Subject-7)', fontsize=20)
        return fig

    def plot_row(self, slices: List, axes: Tuple[Any, Any, Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            imgs = [img for img in slice_]
            imgs = np.concatenate(imgs, axis=1)

            axis.imshow(imgs, cmap="bone", alpha=0.8, vmin=0, vmax=255)
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])


def generate_fig(
    ratio_ec_data: Union[ndarray, ndarray],
    ratio_diff_ec_data: Union[ndarray, ndarray],
    ratio_sign_ec_data: Union[ndarray, ndarray],
    ratio_diff_sign_ec_data: Union[ndarray, ndarray],
    inter_union_vox_ec_data: Union[ndarray, ndarray],
) -> None:
    brainSlice = BrainSlices(
        ratio_ec_data,
        ratio_diff_ec_data,
        ratio_sign_ec_data,
        ratio_diff_sign_ec_data,
        inter_union_vox_ec_data
    )

    fig = brainSlice.plot()

    filename = f"ec_method_image_sub_7.png"
    # outfile = PARAMETRIC_MAP / filename
    fig.savefig(filename)
    fig.savefig('ec_method_image_sub_7.pdf', dpi=120, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    with np.load("/home/mostafiz/Desktop/rUnet_CC/test_prediction/all_method_ec_subject_6.npz") as data:
        ratio_ec_data = data['ratio']
        ratio_diff_ec_data = data['ratio_diff']
        ratio_sign_ec_data = data['ratio_sign']
        ratio_diff_sign_ec_data = data['ratio_diff_sign']
        inter_union_vox_ec_data = data['inter_union_vox']
    generate_fig(ratio_ec_data, ratio_diff_ec_data, ratio_sign_ec_data, ratio_diff_sign_ec_data, inter_union_vox_ec_data)