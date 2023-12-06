import os.path as path
import pickle
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from . import data


class _BaseModule(pl.LightningModule):
    def __init__(self, loss_weight: torch.Tensor | None) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(weight=loss_weight)

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
        estim = np.empty((0, len(data.STATE)), dtype=np.float32)
        truth = np.empty(0, dtype=np.int32)
        for o in self.test_outputs:
            img = np.vstack((img, o[0].cpu().numpy()))
            estim = np.vstack((estim, o[1].cpu().numpy()))
            truth = np.hstack((truth, o[2].squeeze().cpu().numpy()))

        with open(path.join(self.logger.log_dir, "test_outputs.pkl"), mode="wb") as f:
            pickle.dump((img, estim, truth), f)

class CNN3(_BaseModule):
    def __init__(self, param: dict[str, int], loss_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(loss_weight)

        self.save_hyperparameters(param)

        self.bn_1, self.bn_2, self.bn_3 = nn.BatchNorm2d(param["conv_ch_1"]), nn.BatchNorm2d(param["conv_ch_2"]), nn.BatchNorm2d(param["conv_ch_3"])
        self.conv_1 = nn.Conv2d(3, param["conv_ch_1"], param["conv_ks_1"])
        self.conv_2 = nn.Conv2d(param["conv_ch_1"], param["conv_ch_2"], param["conv_ks_2"])
        self.conv_3 = nn.Conv2d(param["conv_ch_2"], param["conv_ch_3"], param["conv_ks_3"])
        self.fc = nn.Linear(((67 - param["conv_ks_1"] - param["conv_ks_2"] - param["conv_ks_3"])) ** 2 * param["conv_ch_3"], len(data.STATE))

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # (batch, channel, height, width) -> (batch, class) or (channel, height, width) -> (1, class)
        if len(input.shape) == 3:
            input = input.unsqueeze(0)

        hidden = F.dropout(F.relu(self.conv_1(input)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_2(hidden)), training=self.training)
        hidden = F.dropout(F.relu(self.conv_3(hidden)), training=self.training)
        output = self.fc(hidden.flatten(start_dim=1))

        return output
