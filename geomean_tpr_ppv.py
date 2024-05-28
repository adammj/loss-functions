# Copyright (C) 2024 Adam M. Jones
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


""" This loss is the geometric mean of the TPR (True Positive Rate),
    and PPV (Positive Predictive Value).
"""


import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss

DEFAULT_EPSILON = 0.0001


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

        # sanity checks
        assert class_count >= 2
        assert confusion.size(0) == class_count
        assert confusion.size(0) == confusion.size(1)
        assert len(torch.where(confusion < 0)[0]) == 0

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

        # scale back up (necessary for SGD to work at all)
        # the previous epsilon is now subtracted out
        final_loss = 1.0 - (intermediate.pow(1.0 / (2.0 * class_count)) - epsilon)

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

        return self.calculate_loss(overall_confusion, self.class_count, self.epsilon)
