# monkey_caput
Code used in fungi image analysis, supervised and unsupervised. Effort described in Towards Data Science, see here and here.

The fungi image data is loaded and pre-procssed in `fungiimg.py` in which the DataSet class is created. The image classification template models are loaded and modified in `model_init.py`. The supervised training of the model takes place in `runner.py`.

The auto-encoder is defined in `ae_deep.py`. Utilities for the clustering is found in `cluster_utils.py`. The training of the auto-encoder as well as the clustering of its code are found in `runner_ae.py`.
