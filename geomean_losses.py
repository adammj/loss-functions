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

import collections
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module


class GeomeanLoss(Module):
    """Loss function computes the geometric mean of the given statistic.

    kappa: class-wise Cohen's kappas
        The loss value will be between 0.0 and 2.0

    tprppv: class-wise True Positive Rates (TPR) and the Positive Predictive Values (PPV)
        The loss value will be between 0.0 and 1.0

    mcc: class-wise Matthews correlation coefficients
        The loss value will be between 0.0 and 2.0"""

    def __init__(self, class_count: int, statistic: str, lower_bound: float = -1.0):
        super().__init__()  # type: ignore

        if class_count < 2:
            raise ValueError("class_count should be >= 2")
        if statistic not in ["kappa", "tprppv", "mcc"]:
            raise ValueError("statistic should be 'kappa', 'tprppv', or 'mcc'")

        # main parameters
        self.class_count = class_count
        self.statistic = statistic

        # for lower_bound adjustment and tracking (only applicable for kappa and mcc)
        self.lower_bound: float = lower_bound
        self.last_min_stat_list: list[float] = collections.deque(maxlen=100)  # type: ignore
        self.last_min_stat: float = 0.0

        # logging for external use
        self.last_confusion: Tensor = torch.zeros(class_count, class_count)
        self.last_loss: float = 0.0

    def forward(
        self,
        input: Tensor,
        target_classes: Tensor,
        weights: Tensor | None,
        auto_lower_bound_adjustment: bool = False,
    ) -> Tensor:
        """pytorch forward call

        If statistic is 'kappa' or 'mcc' and auto_lower_bound_adjustment == True, then
        the function will try to slowly raise the lower_bound from -1.0 to 0.0. The
        adjustment is a one-way ratchet, that will stop at 0.0.
        This will allow the geometric mean to more appropriately weight the positive
        values, while allowing for negative values early in the training.
        """

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

            if auto_lower_bound_adjustment:
                # track the recent last_min_stat values
                self.last_min_stat_list.append(self.last_min_stat)

                # if appropriate, then adjust the lower_bound
                if len(self.last_min_stat_list) >= 10 and self.lower_bound < 0.0:
                    recent_min = min(self.last_min_stat_list) - 0.05
                    if (self.lower_bound + 0.01) < recent_min:
                        # lower_bound can be at most 0.0
                        self.lower_bound = min(self.lower_bound + 0.01, 0.0)

            # even without allowing auto adjustment, the lower bound (lb)
            # must prevent a 0 from being used in the geometric mean
            current_lb = min(self.lower_bound, self.last_min_stat - eps)

            # shift and scale from range [-1, 1] to the range [0, 1]
            # as the geometric does not work with negative values
            class_kappas = (class_kappas - current_lb) / (1.0 - current_lb)

            # calculate the geometric mean of the kappas
            kappa_gm = torch.exp(torch.log(class_kappas).mean())

            # shift and scale the geometric mean back to the range [-1, 1]
            kappa_gm = ((1.0 - current_lb) * kappa_gm) + current_lb

            # calculate the final loss
            final_loss = cast(Tensor, 1.0 - kappa_gm)

        elif self.statistic == "tprppv":
            # compute the two values for each class
            class_tpr = diag / rows
            class_ppv = diag / cols

            # concatenate the TPR and PPV
            concat = torch.cat((class_tpr, class_ppv))

            # store a copy that can be used later
            self.last_min_stat = concat.detach().min().item()

            # calculate the geometric mean of all the agreements
            tprppv_gm = torch.exp(torch.log(concat).mean())

            # calculate the final loss
            final_loss = cast(Tensor, 1.0 - tprppv_gm)

        else:  # self.statistic == "mcc"
            # get all class-wise mccs
            num = diag * (1.0 + (diag - cols - rows)) - (cols - diag) * (rows - diag)
            denom = torch.sqrt(cols * (1.0 - cols) * rows * (1.0 - rows))
            class_mcc = num / denom

            # store a copy that can be used later
            self.last_min_stat = class_mcc.detach().min().item()

            if auto_lower_bound_adjustment:
                # track the recent last_min_stat values
                self.last_min_stat_list.append(self.last_min_stat)

                # if appropriate, then adjust the lower_bound
                if len(self.last_min_stat_list) >= 10 and self.lower_bound < 0.0:
                    recent_min = min(self.last_min_stat_list) - 0.05
                    if (self.lower_bound + 0.01) < recent_min:
                        # lower_bound can be at most 0.0
                        self.lower_bound = min(self.lower_bound + 0.01, 0.0)

            # even without allowing auto adjustment, the lower bound (lb)
            # must prevent a 0 from being used in the geometric mean
            current_lb = min(self.lower_bound, self.last_min_stat - eps)

            # shift and scale from range [-1, 1] to the range [0, 1]
            # as the geometric does not work with negative values
            class_mcc = (class_mcc - current_lb) / (1.0 - current_lb)

            # calculate the geometric mean of the mccs
            mcc_gm = torch.exp(torch.log(class_mcc).mean())

            # shift and scale the geometric mean back to the range [-1, 1]
            mcc_gm = ((1.0 - current_lb) * mcc_gm) + current_lb

            # calculate the final loss
            final_loss = cast(Tensor, 1.0 - mcc_gm)

        # store a copy that can be used later
        self.last_loss = final_loss.detach().item()

        return final_loss
