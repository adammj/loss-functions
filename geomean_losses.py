# Copyright (C) 2024 Adam M. Jones
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""This is classification loss function using the geometric mean of a statistic."""

from typing import cast

import torch
from torch import Tensor
from torch.nn import Module


class GeomeanLoss(Module):
    """
    Loss function computes the geometric mean of the given statistic.

    kappa: class-wise Cohen's kappas

    tprppv: class-wise True Positive Rates (TPR) and the Positive Predictive Values (PPV)

    mcc: class-wise Matthews correlation coefficients

    The loss value will be between 0.0 and 1.0
    """

    def __init__(self, class_count: int, statistic: str):
        super().__init__()  # type: ignore

        if class_count < 2:
            raise ValueError("class_count should be >= 2")
        if statistic not in ["kappa", "tprppv", "mcc"]:
            raise ValueError("statistic should be 'kappa', 'tprppv', or 'mcc'")

        # main parameters
        self.class_count = class_count
        self.statistic = statistic

        # logging for external use
        self.last_confusion: Tensor = torch.zeros(class_count, class_count)
        self.last_min_stat: float = 0.0
        self.last_loss: float = 0.0

    def forward(
        self,
        input: Tensor,
        target_classes: Tensor,
        weights: Tensor | None,
    ) -> Tensor:
        """pytorch forward call"""

        if weights is not None:
            # copy the weights across rows
            weights = weights.unsqueeze(1).expand(-1, self.class_count)

            # multiply the input by the weights
            # this will automatically ignore "bad" samples (with weight = 0)
            # don't modify input in place
            input = input * weights

        # build confusion from each target class
        # this will automatically ignore any padded classes (with target = -1)
        confusion = torch.zeros(
            (self.class_count, self.class_count), dtype=input.dtype, device=input.device
        )
        for i in range(self.class_count):
            confusion[i, :] += input[target_classes == i].sum(0)

        # store a copy that can be used later
        self.last_confusion = confusion.detach().clone()

        # normalize the confusion matrix
        # adjust sum to account for eps being added later to diagonal
        eps = torch.finfo(confusion.dtype).eps
        confusion *= (1.0 - eps * self.class_count) / confusion.sum()

        # minimize numerical issues with empty rows or cols by adding eps to diagonal
        # this will make the confusion sum == 1.0
        confusion += torch.eye(self.class_count, device=confusion.device) * eps

        # get the sums and diagonal
        cols = confusion.sum(0)
        rows = confusion.sum(1)
        diag = confusion.diag()

        if self.statistic == "kappa":
            # get all class-wise kappas
            class_kappas = (
                2.0 * (diag - cols * rows) / (cols + rows - 2.0 * cols * rows)
            )

            # store a copy that can be used later
            self.last_min_stat = class_kappas.detach().min().item()

            # transform from range [-1, 1] to the range [0, 1]
            class_kappas = smooth_transform(class_kappas)

            # calculate the geometric mean of the kappas
            kappa_gm = torch.exp(torch.log(class_kappas).mean())

            # calculate the final loss
            final_loss = cast(Tensor, 1.0 - kappa_gm)

        elif self.statistic == "tprppv":
            # compute the two values for each class
            class_tpr = diag / rows
            class_ppv = diag / cols

            # concatenate the TPR and PPV
            class_concat = torch.cat((class_tpr, class_ppv))

            # store a copy that can be used later
            self.last_min_stat = class_concat.detach().min().item()

            # calculate the geometric mean of all the agreements
            tprppv_gm = torch.exp(torch.log(class_concat).mean())

            # calculate the final loss
            final_loss = cast(Tensor, 1.0 - tprppv_gm)

        else:  # self.statistic == "mcc"
            # get all class-wise mccs
            class_mccs = (diag - cols * rows) / torch.sqrt(
                cast(Tensor, cols * (1.0 - cols) * rows * (1.0 - rows))
            )

            # store a copy that can be used later
            self.last_min_stat = class_mccs.detach().min().item()

            # transform from range [-1, 1] to the range [0, 1]
            class_mccs = smooth_transform(class_mccs)

            # calculate the geometric mean of the mccs
            mcc_gm = torch.exp(torch.log(class_mccs).mean())

            # calculate the final loss
            final_loss = cast(Tensor, 1.0 - mcc_gm)

        # store a copy that can be used later
        self.last_loss = final_loss.detach().item()

        return final_loss


def smooth_transform(x: Tensor, transition: float = 0.066) -> Tensor:
    """
    This is designed to transform the range [-1, 1] to [0, 1], in the case where
    values < 0 are unlikely, but still possible. And where having the positive
    values being left alone is paramount.

    Above the transition, keep original values.
    And below, smoothly transition (both y and y') to a limit of 0.

    A transition of 0.066 gives non-zero outputs for x >= -0.5058 (for float32).
    Since, a kappa or mcc < -0.5 would be exceptionally unlikely, this is probably
    a good compromise.
    """

    y = torch.where(
        x >= transition,
        x,
        transition * (torch.tanh((x - transition) / transition) + 1.0),
    )

    # prevent any (unlikely) zeros from making it to log
    y = torch.where(y > 0, y, torch.finfo(y.dtype).eps)

    return y
