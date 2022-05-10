"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""

from pathlib import Path
from typing import Any, List, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import Figure
from numpy import ndarray
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from utils.const import TEST_FIGURES


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    """Apply a 3D binary mask to a 1-channel, 3D ndarray `img` by creating a 3-channel
    image with masked regions shown in transparent blue."""
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)  # img
    return scaled


def get_logger(logdir: Path) -> TensorBoardLogger:
    return TensorBoardLogger(str(logdir), name="unet")


class BrainSlices:
    def __init__(
        self,
        lightning: LightningModule,
        img: Tensor,
        target: Tensor,
        prediction: Tensor,
        residual: Tensor,
    ):
        # Some code is adapted from: https://github.com/fepegar/unet/blob/master/unet/conv.py#L78
        self.lightning = lightning
        self.input_img: ndarray = (
            img.cpu().detach().numpy().squeeze() if torch.is_tensor(img) else img
        )
        self.target_img: ndarray = (
            target.cpu().detach().numpy().squeeze() if torch.is_tensor(target) else target
        )

        self.predict_img: ndarray = make_imgs(
            prediction.cpu().detach().numpy().squeeze() if torch.is_tensor(prediction) else prediction
        )

        self.residual_img: ndarray = make_imgs(
            residual.cpu().detach().numpy().squeeze() if torch.is_tensor(residual) else residual
        )


        if self.lightning.hparams.task == "longitudinal":  # type: ignore
            if len(self.input_img.shape) == 3:
                si, sj, sk = self.input_img.shape
                i = si // 2
                j = sj // 2
                k = sk // 2
                self.slices = [
                    self.get_slice(self.input_img, i, j, k),
                    self.get_slice(self.target_img, i, j, k),
                    self.get_slice(self.predict_img, i, j, k),
                    self.get_slice(self.residual_img, i, j, k),
                ]
            else:
                si, sj, sk = self.input_img.shape[1:]
                i = si // 2
                j = sj // 2
                k = sk // 2
                if self.input_img.shape[0] == 2:
                    self.slices = [
                        self.get_slice(self.input_img[0], i, j, k),
                        self.get_slice(self.input_img[1], i, j, k),
                        self.get_slice(self.target_img, i, j, k),
                        self.get_slice(self.predict_img, i, j, k),
                        self.get_slice(self.residual_img, i, j, k),
                    ]
                elif self.input_img.shape[0] == 3:
                    self.slices = [
                        self.get_slice(self.input_img[0], i, j, k),
                        self.get_slice(self.input_img[1], i, j, k),
                        self.get_slice(self.input_img[2], i, j, k),
                        self.get_slice(self.target_img, i, j, k),
                        self.get_slice(self.predict_img, i, j, k),
                        self.get_slice(self.residual_img, i, j, k),
                    ]
        self.title = [
            "Input image: M12",
            "Input image: M06",
            "Input image: SC",
            "Target image: M24",
            "Predict image",
            "Residual image",
        ]
        self.shape = np.array(self.input_img.shape)

    def get_slice(self, input: ndarray, i: int, j: int, k: int) -> List[Tuple[ndarray, ...]]:
        return [
            (np.flipud(input[i // 2, ...]), np.flipud(input[i, ...]), np.flipud(input[i + i // 2, ...])),
            (np.flipud(input[:, j // 2, ...]), np.flipud(input[:, j, ...]), np.flipud(input[:, j + j // 2, ...])),
            (np.flipud(input[:, :, k // 2, ...]), np.flipud(input[:, :, k, ...]), np.flipud(input[:, :, k + k // 2, ...])),
        ]

    def plot(self) -> Figure:
        nrows, ncols = len(self.slices), 3  # one row for each slice position

        if nrows == 5:
            fig = plt.figure(figsize=(14, 10))
        if nrows == 6:
            fig = plt.figure(figsize=(13, 10))
        else:
            fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(nrows, ncols)
        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 3])
            ax2 = plt.subplot(gs[i * 3 + 1])
            ax3 = plt.subplot(gs[i * 3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                if self.lightning.hparams.task == "longitudinal":  # type: ignore
                    if i == 0:
                        axis.set_title(self.title[0])
                    elif i == (len(self.slices) - 1):
                        axis.set_title(self.title[-1])
                    elif i == (len(self.slices) - 2):
                        axis.set_title(self.title[-2])
                    elif i == 1:
                        axis.set_title(self.title[1])
                    elif i == 2:
                        axis.set_title(self.title[2])
                    elif i == 3:
                        axis.set_title(self.title[3])
        plt.tight_layout()
        return fig

    def plot_row(self, slices: List, axes: Tuple[Any, Any, Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            imgs = [img for img in slice_]
            imgs = np.concatenate(imgs, axis=1)

            axis.imshow(imgs, cmap="bone", alpha=0.8)
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])

    def log(self, state: str, fig: Figure, loss: float, batch_idx: int) -> None:
        logger = self.lightning.logger
        summary = (
            f"{state}-Epoch:{self.lightning.current_epoch + 1}-batch:{batch_idx}-loss:{loss:0.5e}"
        )
        logger.experiment.add_figure(summary, fig, close=True)
        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)


"""
Actual methods on logger.experiment can be found here!!!
https://pytorch.org/docs/stable/tensorboard.html
"""


def log_all_info(
    module: LightningModule,
    img: Union[Tensor, ndarray],
    target: Union[Tensor, ndarray],
    pred: Union[Tensor, ndarray],
    residual: Union[Tensor, ndarray],
    loss: float,
    batch_idx: int,
    state: str,
    kfold_num: int,
    num_rep: int,
) -> None:
    brainSlice = BrainSlices(
        module, img, target, pred, residual
    )
    fig = brainSlice.plot()

    filename = f"k_{kfold_num}_rep_{num_rep}_bat_{batch_idx}_{state}.png"
    outfile = TEST_FIGURES / filename
    fig.savefig(outfile)
    brainSlice.log(state, fig, loss, batch_idx)
    plt.close()
