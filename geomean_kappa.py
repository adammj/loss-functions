# Copyright (C) 2024  Adam M. Jones
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


""" This loss is the geometric mean of the class-wise Cohen's kappas.
"""


import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss


class GeomeanKappa(_WeightedLoss):
    """Loss function takes the product of the kappa of each class.
    The loss value will be between 0.0 (best) and 1.0 (worst)."""

    def __init__(self, class_count: int):
        super(GeomeanKappa, self).__init__()

        self.class_count = class_count
        self.loss_confusion: Tensor = torch.zeros(class_count, class_count)

    @staticmethod
    def calculate_loss(confusion: Tensor, class_count: int) -> Tensor:
        """static method for just calculating the loss

        broken out so that the function can be called with other confusion matrices"""

        # sanity checks
        assert class_count >= 2
        assert confusion.size(0) == class_count
        assert confusion.size(0) == confusion.size(1)
        assert len(torch.where(confusion < 0)[0]) == 0

        # if the confusion matrix is all zeros, return 1.0
        if confusion.sum() == 0:
            return torch.tensor(1.0)

        # normalize the confusion matrix
        confusion = confusion / confusion.sum()

        # get the sums and diagonal
        cols = confusion.sum(0)
        rows = confusion.sum(1)
        diag = confusion.diag()

        # get all class-wise kappas
        kappas = 2 * (diag - cols * rows) / (cols + rows - 2 * cols * rows)

        # fix any NaNs (happens when pe=1, which means both raters agree but either
        # the class of interest or the collection of other classes is empty)
        kappas[torch.isnan(kappas)] = 1

        # shift and scale the kappas from range [-1, 1] to the range [0, 1]
        kappas = (kappas + 1) / 2

        # check kappas
        assert len(torch.where(kappas < 0)[0]) == 0
        assert len(torch.where(kappas > 1)[0]) == 0

        # calculate the final loss
        final_loss = 1.0 - kappas.prod().pow(1 / class_count)

        return final_loss

    def forward(
        self, input_classes: Tensor, target_classes: Tensor, epoch_weights: Tensor
    ) -> Tensor:
        """pytorch forward call"""

        # copy the weights across rows
        epoch_weights = epoch_weights.unsqueeze(1).repeat(1, self.class_count)

        # multiply the input by the weights
        # this will automatically ignore bad epochs (with weight = 0)
        input_classes = input_classes * epoch_weights

        # build confusion from each target class
        # this will automatically ignore any padded classes (with target = -1)
        overall_confusion_list: list[Tensor] = []
        for i in range(self.class_count):
            overall_confusion_list += [
                input_classes[target_classes == i].sum(0).unsqueeze(0)
            ]
        overall_confusion = torch.cat(overall_confusion_list, 0)

        # store a copy that can be used later.
        self.loss_confusion = overall_confusion.detach().clone()

        return self.calculate_loss(overall_confusion, self.class_count)
