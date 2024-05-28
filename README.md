# Loss functions used in "Expert-level sleep staging using an electrocardiography-only feed-forward neural network"

This repository contains the two loss functions that were created during the development of

> Adam M. Jones, Laurent Itti, Bhavin R. Sheth, "Expert-level sleep staging using an electrocardiography-only feed-forward neural network," Computers in Biology and Medicine, 2024, doi: 10.1016/j.compbiomed.2024.108545

If you find this repository helpful, please cite our work.

The main repository is here: <https://github.com/adammj/ecg-sleep-staging>

---

Both loss functions assume that the final operation of the network is a softmax, which transforms the output into a probability for each of the N classes.

1. **GeomeanKappa**: Geometric Mean of Kappas (**used in paper**).

   This calculates the geometric mean of each of the class-wise kappas.

2. **GeomeanTPRPPV**: Geometric Mean of TPR and PPV.

   This calculates the geometric mean of the True Positive Rates (TPR) and Positive Predictive Values (PPV) for each of the classes.

---

The GeomeanTPRPPV was used for a significant fraction of the hyperparameter search, and performed quite well. However, once I figured out how to calculate the class-wise kappas using a simple equation, I switched to GeomeanKappa. This is because, mathematically, it should be a little closer to the desired metric, Cohen's Kappa (which is the weighted average of the class-wise kappas).

The `calculate_loss` is a separate function, and the `loss_confusion` matrix is stored, to aid some calculations that are done elsewhere in my training code. However, the loss function is a drop-in replacement for any other PyTorch loss function.

---

MIT License

Copyright (C) 2024 Adam M. Jones
