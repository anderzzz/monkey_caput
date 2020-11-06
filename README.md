# monkey_caput
Code used in fungi image analysis, supervised and unsupervised. Effort described in Towards Data Science, see <insert link>.

The fungi image data is loaded and pre-procssed in `fungidata.py` in which the DataSet class is created through a factory method. That includes full images, grid images, with or without ground-truth label or index in dataset. 

Image classification efforts are in files starting with `ic`. The template models for example are loaded in `ic_template_models.py`. The auto-encoder is defined in `ae_deep.py` with a learner class in `ae_learner`. Local Aggregation criterion is found in `cluster_utils` and its learner class in `la_learner`. The training inherits from `_learner`, which can be further generalized.
