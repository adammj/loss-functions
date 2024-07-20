# Copyright (C) 2024 Adam M. Jones
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


""" These are classification loss functions using the geometric mean.
"""


import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss

# used for GeomeanTPRPPV to prevent division by zero or product of zero issues
DEFAULT_EPSILON = 0.0001


class GeomeanKappa(_WeightedLoss):
    """Loss function takes the product of the kappa of each class.
    The loss value will be between 0.0 (best) and 1.0 (worst)."""

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

        # sanity check of kappas
        assert len(torch.where(kappas < 0)[0]) == 0
        assert len(torch.where(kappas > 1)[0]) == 0

        # calculate the final loss
        final_loss = 1.0 - kappas.prod().pow(1 / class_count)

        return final_loss

    def forward(
        self, input: Tensor, target_classes: Tensor, weights: Tensor | None
    ) -> Tensor:
        """pytorch forward call"""

        if weights is not None:
            # copy the weights across rows
            weights = weights.unsqueeze(1).repeat(1, self.class_count)

            # multiply the input by the weights
            # this will automatically ignore "bad" samples (with weight = 0)
            input = input * weights

        # build confusion from each target class
        # this will automatically ignore any padded classes (with target = -1)
        overall_confusion_list: list[Tensor] = []
        for i in range(self.class_count):
            overall_confusion_list += [
                input[target_classes == i].sum(0).unsqueeze(0)
            ]
        overall_confusion = torch.cat(overall_confusion_list, 0)

        # store a copy that can be used later.
        self.loss_confusion = overall_confusion.detach().clone()

        return self.calculate_loss(overall_confusion, self.class_count)


class GeomeanTPRPPV(_WeightedLoss):
    """Loss function takes the product of the True Positive Rates and the
    Positive Predictive Values for each class. It approximates 1-(overall kappa),
    with many fewer operations.

    epsilon is used to prevent both div by zero, and a 0 on any diag element
    causing the loss to jump to 1.
    the final_loss will be between epsilon (best) and 1.0 (worst)"""

    def __init__(self, class_count: int, epsilon: float = DEFAULT_EPSILON):
        super(GeomeanTPRPPV, self).__init__()

        # tiny offset to prevent any issues with zeros
        self.epsilon = epsilon
        self.class_count = class_count
        self.loss_confusion: Tensor = torch.zeros(class_count, class_count)

    @staticmethod
    def calculate_loss(
        confusion: Tensor, class_count: int, epsilon: float = DEFAULT_EPSILON
    ) -> Tensor:
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

        # if the confusion matrix is all zeros, return 1.0
        if confusion.sum() == 0:
            return torch.tensor(1.0)

        # compute the two outputs
        # epsilon is used here to prevent division by zero
        class_tpr = confusion.diag() / (confusion.sum(1) + epsilon)
        class_ppv = confusion.diag() / (confusion.sum(0) + epsilon)

        # combine the TPR and PPV
        # epsilon is used here to prevent any 0 diagonal element from forcing the loss
        # to the maximum value
        intermediate = torch.cat((class_tpr, class_ppv)) + epsilon

        # take the product of all elements
        intermediate = intermediate.prod()

        # calculate the final loss
        final_loss = 1.0 - (intermediate.pow(1.0 / (2.0 * class_count)) - epsilon)

        return final_loss

    def forward(
        self, input: Tensor, target_classes: Tensor, weights: Tensor | None
    ) -> Tensor:
        """pytorch forward call"""

        if weights is not None:
            # copy the weights across rows
            weights = weights.unsqueeze(1).repeat(1, self.class_count)

            # multiply the input by the weights
            # this will automatically ignore "bad" samples (with weight = 0)
            input = input * weights

        # build confusion from each target class
        # this will automatically ignore any padded classes (with target = -1)
        overall_confusion_list: list[Tensor] = []
        for i in range(self.class_count):
            overall_confusion_list += [
                input[target_classes == i].sum(0).unsqueeze(0)
            ]
        overall_confusion = torch.cat(overall_confusion_list, 0)

        # store a copy that can be used later.
        self.loss_confusion = overall_confusion.detach().clone()

        return self.calculate_loss(overall_confusion, self.class_count, self.epsilon)
