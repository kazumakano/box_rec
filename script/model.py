import os.path as path
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from .data import Usage


class _BaseModule(pl.LightningModule):
    def __init__(self, loss_weight: torch.Tensor | None, param: dict[str, float | int]) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(weight=loss_weight)
        self.save_hyperparameters(param)

    def configure_optimizers(self) -> optim.SGD:
        return optim.SGD(self.parameters(), lr=self.hparams["learning_rate"])

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        estim = self(batch[0])
        loss = self.criterion(estim, batch[1])
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor, _: int) -> None:
        estim = self(batch[0])
        loss = self.criterion(estim, batch[1])
        self.log("validation_loss", loss)

    def on_test_start(self) -> None:
        self.test_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def test_step(self, batch: torch.Tensor, _: int) -> None:
        self.test_outputs.append((batch[0], self(batch[0]), batch[1]))

    def on_test_end(self) -> None:
        img = np.empty((0, 3, 64, 64), dtype=np.float32)
        estim = np.empty((0, len(Usage)), dtype=np.float32)
        truth = np.empty(0, dtype=np.int32)
        for o in self.test_outputs:
            img = np.vstack((img, o[0].cpu().numpy()))
            estim = np.vstack((estim, o[1].cpu().numpy()))
            truth = np.hstack((truth, o[2].squeeze().cpu().numpy()))

        np.savez_compressed(path.join(self.logger.log_dir, "test_outputs.npz"), img=img, estim=estim, truth=truth)

class CNN3(_BaseModule):
    def __init__(self, param: dict[str, float | int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight, param)

        self.conv_1 = nn.Conv2d(3, param["conv_ch_1"], param["conv_ks_1"])
        self.conv_2 = nn.Conv2d(param["conv_ch_1"], param["conv_ch_2"], param["conv_ks_2"])
        self.conv_3 = nn.Conv2d(param["conv_ch_2"], param["conv_ch_3"], param["conv_ks_3"])
        self.fc = nn.Linear(((67 - param["conv_ks_1"] - param["conv_ks_2"] - param["conv_ks_3"])) ** 2 * param["conv_ch_3"], len(Usage))

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class) or (channel, height, width) -> (class, )
        hidden = F.dropout(F.relu(self.conv_1(input)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_3(hidden)), training=self.training)
        output = self.fc(hidden.flatten(start_dim=-3))

        return output
