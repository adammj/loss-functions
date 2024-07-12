# Loss functions for imbalanced classification and/or where Cohen's kappa is the metric

This repository contains the two loss functions that were created during the development of

> Adam M. Jones, Laurent Itti, Bhavin R. Sheth, "Expert-level sleep staging using an electrocardiography-only feed-forward neural network," Computers in Biology and Medicine, 2024, doi: 10.1016/j.compbiomed.2024.108545

(Link to paper: <https://doi.org/10.1016/j.compbiomed.2024.108545>.)


If you find this repository helpful, please cite our work.

The main repository for the paper is here: <https://github.com/adammj/ecg-sleep-staging>

---

## Motivation

These loss functions were designed for imbalanced classification problems, where it is not possible to oversample the minority classes or undersample the majority classes (please see the paper for a more thorough explanation of this situation). Furthermore, most classification problems assume that accuracy is the desired metric, and therefore use [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression) as the loss function. However, for our use-case, [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) is the correct metric (which is only loosely correlated with accuracy).

Normally, especially in highly imbalanced data, correctly classifying the majority class(es) will almost always be at the expense of the minority class(es). However, we use the [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean) of the individual class performances (kappa, TPR, or PPV), which are all in the range of [0, 1] (_see below for kappa_). This has the effect of causing the loss function to balance competing ratios, instead of competing counts (which will always disfavor the minority).

Both loss functions assume that the final operation of the network is a softmax, which transforms the output into a probability for each of the N classes.

1. **GeomeanKappa**: Geometric Mean of Kappas (**used in paper**).

   This calculates the geometric mean of each of the class-wise kappas. The class-wise kappas are scaled using `(1 + k)/2`, so that the default kappa range is transformed from [-1, 1] to [0, 1].

   By doing so, it will tend to improve all of the class-wise kappas.

2. **GeomeanTPRPPV**: Geometric Mean of TPR and PPV.

   This calculates the geometric mean of the True Positive Rates ([TPR or sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)) and Positive Predictive Values ([PPV or precision](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)) for each of the classes.

   By doing so, it will tend to both increase the TPR (number of correctly classified instances divided by the total possible instances for reviewer 1 (rows)), and PPV (number of correctly classified instances, divided by all instances that reviewer 2 (columns) used the same class). For example, with imbalanced classes, it will simultaneously work to correctly classify as many of the majority class as possible (minimizing off-diagonal counts in the rows), while minimizing the number of incorrect classifications that occur against the minority class (minimizing off-diagonal counts in the columns).

---

## Comparisons against other loss functions

For our final model, we substituted in several different loss functions in order to compare them against our loss function. We'd like to highlight that for the two functions where Overall kappa is slightly higher (+1%), their minority class (N1) performance is significantly worse (-27%).

The table gives the kappa for each sleep stage and loss function pair.

| Loss function                   | Overall | Wake  | N1    | N2    | N3    | REM   |
| ------------------------------- | ------- | ----- | ----- | ----- | ----- | ----- |
| Geometric Mean of Kappas (ours) | 0.726   | 0.862 | 0.373 | 0.671 | 0.703 | 0.805 |
| Cross-entropy                   | 0.734   | 0.867 | 0.274 | 0.682 | 0.699 | 0.805 |
| Cross-entropy (weighted)        | 0.669   | 0.845 | 0.332 | 0.583 | 0.677 | 0.786 |
| Focal loss                      | 0.732   | 0.862 | 0.297 | 0.679 | 0.703 | 0.801 |
| Cohen’s kappa (overall)         | 0.720   | 0.854 | 0.000 | 0.669 | 0.697 | 0.795 |
| Ratio of ours to best           | 99%     | 99%   | 100%  | 98%   | 100%  | 100%  |

---

## Additional details

The GeomeanTPRPPV was used for a significant fraction of the hyperparameter search, and performed quite well. However, once I figured out how to calculate the class-wise kappas using a simple equation, I switched to GeomeanKappa. This is because, mathematically, it should be a little closer to the desired metric, [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) (which is the weighted average of the class-wise kappas).

The `calculate_loss` is a separate function, and the `loss_confusion` matrix is stored, to aid some calculations that are done elsewhere in my training code. However, the loss function is a drop-in replacement for any other PyTorch loss function.

---

MIT License

Copyright (C) 2024 Adam M. Jones
