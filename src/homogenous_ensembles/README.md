# HOMOGENOUS ENSEMBLES

Ensembling is a technique where multiple models are used to derive a single output.
In some cases, models of differing architectures and instantiations can be made into ensembles.
In the case of homogenous ensembles, all models are separate instantiations of the SAME model.

The code in this directory relies primarily on [TorchEnsemble](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch), with hacks to make it play nice* with pytorch-lightning and aim logger. Torchensemble only supports making ensembles of models with identical architectures and hparam settings. It's amalgamation with the functionality provided by pytorch-lightning (validation, metrics, logging, callbacks) should be considered experimental. 

To read more on TorchEnsemble, such as the ensembling techniques it can perform, you can [read the docs](https://ensemble-pytorch.readthedocs.io/en/latest/introduction.html).

`train_ensemble.py` has many of the same parameter flags as the main `train.py`, and some additional params that are ensemble-specific. The (homogenous model) ensembling methods available to use are:

* Fusion
* Voting
* Bagging
* Boosting
* Snapshot
* Adversarial
* FastGeometric


