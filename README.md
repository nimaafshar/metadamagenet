# MetaDamageNet

#### Using Deep Learning To Identify And Classify Damage In Aerial Imagery

This project is my bachelors thesis at AmirKabir University of Technology under supervision of Dr.Amin Gheibi.
Some ideas of this project is borrowed from
[xview2 first place solution](https://github.com/vdurnov/xview2_1st_place_solution)
repository. I used that repository as a baseline and refactored its code.
Thus, this code covers models and experiments of the mentioned repo and
contributes more research into the same problem of damage assessment in aerial imagery.

## Usage

### environment setup

```bash
git clone https://github.com/nimaafshar/metadamagenet.git
cd metadamagenet/
pip install -r requirements.txt
```

### examples

- [create segmentation masks from json labels](./create_masks.py)
- [`Resnet34Unet` training and tuning](./example_resnet34.py)
- [`SeResnext50Unet` training and tuning](./example_seresnext50.py)
- [`Dpn92Unet` training and tuning](./example_dpn92.py)
- [`SeNet154Unet` training and tuning](./example_dpn92.py)

## Data

### Structure and

### Data Loading and Datasets

### Augmentations

## Methodology

Example Usage:

```python
from metadamagenet.models import Localizer
from metadamagenet.models.unet import EfficientUnetB0


# define localizer of unet
class EfficientUnetB0Localizer(Localizer[EfficientUnetB0]): pass


# load pretrained model
pretrained_model = EfficientUnetB0Localizer.from_pretrained(version='00', seed=0)

# load an empty model
empty_model = EfficientUnetB0Localizer()

# load a model from pretrained unet
unet: EfficientUnetB0  # some pretrained unet
model_with_pretrained_unet = EfficientUnetB0Localizer(unet)

# load an empty unet
empty_unet = EfficientUnetB0()

# load a unet with pretrained backbone
unet_with_pretrained_backbone = EfficientUnetB0(pretrained_backbone=True)
```

### General Architecture

As shown in the figure below, building-localization models consist of a feature extractor (a U-net or a SegFormer) and a
classifier module.
The feature extractor extracts helpful features from the input image;
then, the classifier module predicts a value of 0 or 1 for each pixel,
indicating whether this pixel belongs to a building or not.
The feature extractor module extracts the same features from pre-disaster and post-disaster images in the classification
models.
In these models, the classifier module predicts a class between 0 and 4 for each pixel.
The value 0 indicates that this pixel belongs to no building;
values 1-4 mean that this pixel belongs to a building and show the damage level in that pixel.
The classifier module learns a distance function between pre-disaster and post-disaster images
because the damage level of each facility can be determined by comparing it in the pre- and post-disaster images.
In many samples, the post-disaster image has a minor shift compared to the pre-disaster image;
the segmentation masks are created based on the location of buildings in the pre-disaster image.
This shift is an issue the model has to overcome. For models that share a joint feature extractor,
we can initialize the feature extractor module in the classification model with the localization model's feature
extractor.
Since we do not use the localization model directly for damage assessment, training of the localization model can be
seen as a pre-training stage for the classification model.

![General Architecture](./res/arch.png)

### U-Models

Some models in this project use a U-net module as the feature extractor and a superficial 2d Convolutional Layer as the
classifier.
We call them u-models. Their feature extractor module is a u-net with five encoder and five decoder modules.
Encoder models are usually a part of a general feature extractor like Resnet-34.
in the forward pass of each image through an encoder module, the number of channels may or may not change.
Still, the height and width of the image are divided by two.
Usually, the five encoder modules combined include all layers of a general feature extractor model (like Resnet34)
except the classification layer.
Each decoder module combines the output of the previous decoder module and the respective encoder module.
For example, encoder module 2 combines the output of decoder module 1 and encoder module 3.
They form a U-like structure, as shown in the figure below.

![Unet](./res/unet-architecture.png)

#### Decoder Modules

There are two variants of decoder modules:
The standard decoder module and the SCSE decoder module.
The standard decoder module applies a 2d convolution and a Relu activation to the input from the previous decoder.
Then it concatenates the result with the input from the respective encoder module and applies another 2d convolution and
ReLU activation.
SCSE decoder module works the same way but, in the end,
uses a "Concurrent Spatial and Channel Squeeze & Excitation" module on the result.
This SCSE module is supposed to help the model focus on the image's more critical regions and channels.
Decoder modules in the forked repository don't use batch normalization between the convolution and the activation.
We added this layer to the decoder modules to prevent gradient exploding and make them more stable.

![Decoder Modules](./res/decoder.png)

### Backbone

We pick encoder modules of U-net models from a general feature extractor model called the backbone network.
The choice of the backbone network is the most crucial point in the performance of a U-net model.
Plus, most of the parameters of a U-net model are of its backbone network.
Thus, the choice of the backbone network significantly impacts its size and performance.
The forked repository used *Resnet34*, *Dual Path Network 92*, *SeResnext50 (32x4d)*, and *SeNet154* as the backbone
network.
We used *EfficientNet B0* and *EfficientNet B4* (both standard and *Wide-SE* versions) as the backbone network, creating
new U-models called EfficientUnets.
EfficientNets have shown excellent results on the ImageNet dataset, so they are good feature extractors.
They are also relatively small in size. These two features make them perfect choices for a backbone network.

We listed all the used U-net models and their attributes in the table below.

<table>
  <tr>
    <td rowspan="1" colspan="2">model</td>
    <td rowspan="2" colspan="1">#params</td>
    <td rowspan="2" colspan="1">Batch Normalization</td>
    <td rowspan="2" colspan="1">DecoderType</td>
  </tr>
  <tr>
    <td colspan="1">name</td>
    <td colspan="1">backbone</td>
  </tr>
  <tr>
    <td>Resnet34Unet</td>
    <td>resnet_34</td>
    <td>25,728,112</td>
    <td> No </td>
    <td>Standard</td>
  </tr>
  <tr>
    <td>SeResnext50Unet</td>
    <td>se_resnext50_32x4d</td>
    <td>34,559,728</td>
    <td>No</td>
    <td>Standard</td>
  </tr>
  <tr>
    <td>Dpn92Unet</td>
    <td>dpn_92</td>
    <td>47,408,735</td>
    <td>No</td>
    <td>SCSE - concat</td>
  </tr>
  <tr>
    <td>SeNet154Unet</td>
    <td>senet_154</td>
    <td>124,874,656</td>
    <td>No</td>
    <td>Standard</td>
  </tr>
  <tr>
    <td>EfficientUnetB0</td>
    <td rowspan="2">efficientnet_b0</td>
    <td>6,884,876</td>
    <td>Yes</td>
    <td>Standard</td>
  </tr>
  <tr>
    <td>EfficientUnetB0SCSE</td>
    <td>6,903,860</td>
    <td>Yes</td>
    <td>SCSE - no concat</td>
  </tr>
  <tr>
    <td>EfficientUnetWideSEB0</td>
    <td>efficientnet_widese_b0</td>
    <td>10,020,176</td>
    <td>Yes</td>
    <td>Standard</td>
  </tr>
  <tr>
    <td>EfficientUnetB4</td>
    <td rowspan="2">efficientnet_b0</td>
    <td>20,573,144</td>
    <td>Yes</td>
    <td>Standard</td>
  </tr>
  <tr>
    <td>EfficientUnetB4SCSE</td>
    <td>20,592,128</td>
    <td>Yes</td>
    <td>SCSE- no concat</td>
  </tr>
</table>

### Vision Transformer


### Meta-Learning
In meta-learning, a general problem, such as classifying different images (in the *ImageNet* dataset) or classifying 
different letters (in the *Omniglot* dataset), is seen as a distribution of tasks. In this approach, tasks are generally 
the same problem (like classifying letters) but vary in some parameters (like the script letters belong to). 
We can take a similar approach to our problem. We can view building detection and damage level classification as the 
general problem and the disaster type (like a flood, hurricane, or wildfire) and the environment of the disaster 
(like a desert, forest, or urban area) as the varying factor. In distance-learning methods, the distance function 
returns a distance between the query sample and each class's sample. Then the query sample is classified into the 
class with minimum distance. These methods are helpful when we have a high number of classes. However, in our case, 
the number of classes is fixed. Thus, we used a model-agnostic approach. Model agnostic meta-learning algorithms find a 
set of parameters for the model that can be adapted to a new task by training with very few samples. 
We used the MAML algorithm and considered every different disaster a task. 
Since the MAML algorithm consumes lots of memory, and the consumed memory is relative to the model size, 
we have used models based on *EfficientUnetB0* for this section.

### Training Setup

### Loss Functions

Example Usage:

```python
from metadamagenet.losses import WeightedSum, BinaryDiceLoss, BinaryFocalLoss

WeightedSum(
    (BinaryDiceLoss(), 1.0),
    (BinaryFocalLoss(alpha=0.7, gamma=2., reduction='mean'), 6.0)
)
```

Both Building Localization and Damage Classification are semantic segmentation tasks.
Because, in both problems, the model's purpose is classification at the pixel level.
We have used a combination of multiple segmentation losses for all models.
[Here](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions),
you can find a comprehensive comparison between popular loss functions for semantic segmentation.

Focal, Dice, and Lovasz-sigmoid Loss are loss functions used in the training localization models.
For Classification models, we tried Focal, Dice, Lovasz-Softmax Loss, Log-Cosh-Dice, and, Cross-entropy Loss.

**Focal Loss**

:page_facing_up: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

$$
FL(p_t) = -\alpha_t(1- p_t)\gamma log(p_t).
$$

Focal Loss's usage is to make the model focus on hard-to-classify examples by increasing their loss value. We used it
because the target distribution was highly skewed. In the building localization task, the number of pixels containing
buildings was far less than the background pixels. In the damage classification task, too, undamaged building samples
formed more than 80 percent of the total samples.

**Dice Loss**

$$
Dice\space Loss(p,t) = 1 - dice(p,t)
$$

Where $dice$, $p$ and $t$ stand for *dice coefficient*, *predictions* and *target values* respectively.

$$
dice(A,B) = 2\frac{ A\cap B}{A + B}
$$

Dice loss is calculated globally over each mini-batch. For multiclass cases, the loss value of each class (channel) is
calculated individually, and their average is used as the final loss. Two activation functions can be applied to model
outputs before calculating dice loss: sigmoid and softmax. Softmax makes the denominator of the final loss function
constant and thus has less effect on the model's training though it makes better sense.

- [example argument about correct implementation of softmax-dice-loss](https://github.com/keras-team/keras/issues/9395#issuecomment-379276452)

**Cross Entropy Loss**

$$
-\sum_{c=1}^My_{o,c}\log(p_{o,c})
$$

Since we used sigmoid-dice-loss for multiclass damage classification, cross-entropy loss helped the model assign only
one class to each pixel. It solely is a good loss function for semantic segmentation tasks.

## Evaluation

Example Usage:

```python
from torch import Tensor
from metadamagenet.metrics import DamageLocalizationMetric, DamageClassificationMetric

evaluator = 0.2 * DamageLocalizationMetric() + 0.8 * DamageClassificationMetric()

preds: Tensor
targets: Tensor
score = evaluator(preds, targets)
```

One of the most popular evaluation metrics for classifiers is the f1-score; because it accounts for precision and recall
simultaneously. The macro version of the f1-score is a good evaluation measure for imbalanced datasets. The
xview2-scoring repository describes what variation of f1-score to use for this problem's scoring. We adapted their
evaluation metrics. However, we implemented these metrics as a metric for the torchmetrics repository. It performs
better than computing metrics in NumPy and provides an easy-to-use API.

- The dice score is a set similarity measure that equals the f1-score.

$$
Dice(P,Q) = 2. \frac{P \cap Q}{P+Q}
$$

$$
F1(P,Q) = \frac{2TP}{2TP + FP + FN}
$$

### Localization Models Scoring

The localization score defines as a globally calculated binary f1-score. Sample-wise calculation means calculating the
score on each sample (image); then averaging sample scores to get the final score. In global calculation, we use the sum
of true positives, true negatives, false positives, and false negatives across all samples to calculate the metric.

The localization score is a binary f1-score, which means class zero (no-building/background) is considered negative, and
class one (building) is considered positive. Since we only care about detecting buildings from the background,
micro-average is applied too.

### Classification Models Scoring

The classification score consists of a weighted sum of 2 scores: the localization score and the damage classification
score. Classification models a label of zero to four for each pixel, indicating no-building, no damage, minor damage,
major damage, and destroyed, respectively. Since one to four label values show that this pixel belongs to a building, we
calculate the localization score after converting all values above zero to one. This score determines how good the model
is at segmenting buildings. We defined the damage classification score as the harmonic mean of the globally computed
f1-score for each class from one to four. We calculate the f1-score of each class separately, then use their harmonic
mean to give each damage level equal importance. Here we prefer the harmonic mean to the arithmetic mean because
different classes do not have equal support. We compute the damage classification score only on the pixels that have one
to four label values in reality. This way, we remove the effect of the models' localization performance from the damage
classification score. Hence, these two metrics represent the models' performance in two disparate aspects.

$$
score = 0.3 \times F1_{LOC} + 0.7 \times F1_{DC}
$$

$$
F1_{DC} = 4/(\frac{1}{F1_1 + \epsilon} + \frac{1}{F1_2 + \epsilon} + \frac{1}{F1_3 + \epsilon} + \frac{1}{F1_4 +
\epsilon})
$$

### Training Results

- EfficientUnet Localization Models have shown that they train better without the usage of focal loss

complete results are available at [results.md](./results.md)

## Conclusion and Acknowledgments

## References

# Data

# Data Cleaning Techniques

Training masks generated using json files, "un-classified" type treated as "no-damage" (create_masks.py). "masks"
folders will be created in "train" and "tier3" folders.

The problem with different nadirs and small shifts between "pre" and "post" images solved on models level:

- First, localization models trained using only "pre" images to ignore this additional noise from "post" images. Simple
  UNet-like segmentation Encoder-Decoder Neural Network architectures used here.
- Then, already pretrained localization models converted to classification Siamese Neural Network. So, "pre" and "post"
  images shared common weights from localization model and the features from the last Decoder layer concatenated to
  predict damage level for each pixel. This allowed Neural Network to look at "pre" and "post" separately in the same
  way and helped to ignore these shifts and different nadirs as well.
- Morphological dilation with 5*5 kernel applied to classification masks. Dilated masks made predictions more "bold" -
  this improved accuracy on borders and also helped with shifts and nadirs.

# Data Processing Techniques

Models trained on different crops sizes from (448, 448) for heavy encoder to (736, 736) for light encoder.
Augmentations used for training:

- Flip (often)
- Rotation (often)
- Scale (often)
- Color shifts (rare)
- Clahe / Blur / Noise (rare)
- Saturation / Brightness / Contrast (rare)
- ElasticTransformation (rare)

Inference goes on full image size (1024, 1024) with 4 simple test-time augmentations (original, filp left-right, flip
up-down, rotation to 180).

# Details on Modeling Tools and Techniques

trained with Train/Validation random split 90%/10% with fixed seeds (3 folds). Only checkpoints from epochs
with best validation score used.

Localization models trained on "pre" images, "post" images used in very rare cases as additional augmentation.

Localization training parameters:
Loss: Dice + Focal
Validation metric: Dice
Optimizer: AdamW

Classification models initialized using weights from corresponding localization model and fold number. They are Siamese
Neural Networks with whole localization model shared between "pre" and "post" input images. Features from last Decoder
layer combined for classification. Pretrained weights are not frozen.
Using pretrained weights from localization models allowed to train classification models much faster and to have better
accuracy. Features from "pre" and "post" images connected at the very end of the Decoder in bottleneck part, this
helping not to over-fit and get better generalizing model.

Classification training parameters:
Loss: Dice + Focal + CrossEntropyLoss. Larger coefficient for CrossEntropyLoss and 2-4 damage classes.
Validation metric: competition metric
Optimizer: AdamW
Sampling: classes 2-4 sampled 2 times to give them more attention.

Almost all checkpoints finally finetuned on full train data for few epoches using low learning rate and less
augmentations.

Predictions averaged with equal coefficients for both localization and classification models separately.

Different thresholds for localization used for damaged and undamaged classes (lower for damaged).

# Conclusion and Acknowledgments

Thank you to xView2 team for creating and releasing this amazing dataset and opportunity to invent a solution that can
help to response to the global natural disasters faster. I really hope it will be usefull and the idea will be improved
further.

## References

- Competition and Dataset: [Xview2 org.](https://www.xview2.org)
- [Xview2 First Place Solution](https://github.com/vdurnov/xview2_1st_place_solution)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Cadene/Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch)
