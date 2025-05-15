# Copyright (C) 2024 Adam M. Jones
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""These are classification loss functions using the geometric mean."""

import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss


class GeomeanKappa(_WeightedLoss):
    """Loss function takes the geometric mean of all class-wise kappas.

    The loss value will approximate 1.0 minus the overall kappa,
    and will be between 0.0 (best) and 2.0 (worst)."""

    def __init__(self, class_count: int):
        super(GeomeanKappa, self).__init__()

        self.class_count = class_count
        self.loss_confusion: Tensor = torch.zeros(class_count, class_count)

    @staticmethod
    def calculate_loss(confusion: Tensor, class_count: int) -> Tensor:
        """static method for just calculating the loss"""

        # input checks
        if class_count < 2:
            raise ValueError("class_count must be >= 2")
        if confusion.size(0) != class_count:
            raise ValueError("each confusion dimension must equal the class_count")
        if confusion.size(0) != confusion.size(1):
            raise ValueError("confusion shape must be square")
        if len(torch.where(confusion < 0)[0]) != 0:
            raise ValueError("confusion should contain no negative elements")
        if confusion.sum() <= 0:
            raise ValueError("confusion sum should be > 0")

        # normalize the confusion matrix
        # adjust sum to account for eps being added later to diagonal
        eps = torch.finfo(confusion.dtype).eps
        confusion *= (1.0 - eps * class_count) / confusion.sum()

        # minimize numerical issues with empty rows or cols by adding eps to diagonal
        confusion += torch.eye(class_count, device=confusion.device) * eps

        # get the sums and diagonal
        cols = confusion.sum(0)
        rows = confusion.sum(1)
        diag = confusion.diag()

        # get all class-wise kappas
        kappas = 2.0 * (diag - cols * rows) / (cols + rows - 2.0 * cols * rows)

        # shift and scale the kappas from range [-1, 1] to the range [0, 1]
        kappas = (kappas + 1.0) / 2.0

        # calculate the geometric mean of the kappas
        kappas_gm = kappas.prod().pow(1.0 / class_count)

        # shift and scale the geometric mean back to the range [-1, 1]
        kappas_gm = (2.0 * kappas_gm) - 1.0

        # calculate the final loss
        final_loss = 1.0 - kappas_gm

        return final_loss

    def forward(
        self, input: Tensor, target_classes: Tensor, weights: Tensor | None
    ) -> Tensor:
        """pytorch forward call"""

        if weights is not None:
            # copy the weights across rows
            weights = weights.unsqueeze(1).expand(-1, self.class_count)

            # multiply the input by the weights
            # this will automatically ignore "bad" samples (with weight = 0)
            input *= weights

        # build confusion from each target class
        # this will automatically ignore any padded classes (with target = -1)
        overall_confusion = torch.zeros(
            (self.class_count, self.class_count), dtype=input.dtype, device=input.device
        )
        for i in range(self.class_count):
            overall_confusion[i, :] += input[target_classes == i].sum(0)

        # store a copy that can be used later.
        self.loss_confusion = overall_confusion.detach().clone()

        return self.calculate_loss(overall_confusion, self.class_count)


class GeomeanTPRPPV(_WeightedLoss):
    """Loss function takes the geometric mean of the True Positive Rates (TPR) and the
    Positive Predictive Values (PPV) for all classes.

    It approximates 1-(overall kappa), with fewer operations.

    The loss value will be between 0.0 (best) and 1.0 (worst)."""

    def __init__(self, class_count: int):
        super(GeomeanTPRPPV, self).__init__()

        # tiny offset to prevent any issues with zeros
        self.class_count = class_count
        self.loss_confusion: Tensor = torch.zeros(class_count, class_count)

    @staticmethod
    def calculate_loss(confusion: Tensor, class_count: int) -> Tensor:
        """static method for just calculating the loss"""

        # input checks
        if class_count < 2:
            raise ValueError("class_count must be >= 2")
        if confusion.size(0) != class_count:
            raise ValueError("each confusion dimension must equal the class_count")
        if confusion.size(0) != confusion.size(1):
            raise ValueError("confusion shape must be square")
        if len(torch.where(confusion < 0)[0]) != 0:
            raise ValueError("confusion should contain no negative elements")
        if confusion.sum() <= 0:
            raise ValueError("confusion sum should be > 0")

        # normalize the confusion matrix
        # adjust sum to account for eps being added later to diagonal
        eps = torch.finfo(confusion.dtype).eps
        confusion *= (1.0 - eps * class_count) / confusion.sum()

        # minimize numerical issues with empty rows or cols by adding eps to diagonal
        confusion += torch.eye(class_count, device=confusion.device) * eps

        # get the sums and diagonal
        cols = confusion.sum(0)
        rows = confusion.sum(1)
        diag = confusion.diag()

        # compute the two outputs
        class_tpr = diag / rows
        class_ppv = diag / cols

        # combine the TPR and PPV
        intermediate = torch.cat((class_tpr, class_ppv))

        # calculate the geometric mean of the agreements
        agreement_gm = intermediate.prod().pow(1.0 / (2.0 * class_count))

        # calculate the final loss
        final_loss = 1.0 - agreement_gm

        return final_loss

    def forward(
        self, input: Tensor, target_classes: Tensor, weights: Tensor | None
    ) -> Tensor:
        """pytorch forward call"""

        if weights is not None:
            # copy the weights across rows
            weights = weights.unsqueeze(1).expand(-1, self.class_count)

            # multiply the input by the weights
            # this will automatically ignore "bad" samples (with weight = 0)
            input *= weights

        # build confusion from each target class
        # this will automatically ignore any padded classes (with target = -1)
        overall_confusion = torch.zeros(
            (self.class_count, self.class_count), dtype=input.dtype, device=input.device
        )
        for i in range(self.class_count):
            overall_confusion[i, :] += input[target_classes == i].sum(0)

        # store a copy that can be used later.
        self.loss_confusion = overall_confusion.detach().clone()

        return self.calculate_loss(overall_confusion, self.class_count)
