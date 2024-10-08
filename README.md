# MetaDamageNet

#### Using Deep Learning To Identify And Classify Building Damage 

This project is my bachelor's thesis at AmirKabir University of Technology.
Some ideas for this project are borrowed from
[xview2 first place solution](https://github.com/vdurnov/xview2_1st_place_solution) [^first_place_solution] repository. 
I used the mentioned repository as a baseline and refactored its code.
Thus, this project covers models and experiments of the mentioned repo and
contributes more to the same problem of damage assessment for buildings.

**Environment Setup**

```bash
git clone https://github.com/nimaafshar/metadamagenet.git
cd metadamagenet/
pip install -r requirements.txt
```

**Examples**

- [create segmentation masks from JSON labels](./create_masks.py)
- [`Resnet34Unet` training and tuning](./example_resnet34.py)
- [`SeResnext50Unet` training and tuning](./example_seresnext50.py)
- [`Dpn92Unet` training and tuning](./example_dpn92.py)
- [`SeNet154Unet` training and tuning](./example_dpn92.py)

### Table Of Contents

- [Dataset](#dataset)
- [Problem Definition](#problem-defenition)
- [Data Augmentations](#data-augmentations)
- [Methodology](#methodology)
    - [General Architecture](#general-architecture)
    - [U-Models](#u-models)
        - [Decoder Modules](#decoder-modules)
        - [Backbone](#backbone)
    - [Meta-Learning](#meta-learning)
    - [Vision Transformer](#vision-transformer)
    - [Training Setup](#training-setup)
    - [Loss Functions](#loss-functions)
- [Evaluation](#evaluation)
    - [Localization Models Scoring](#localization-models-scoring)
    - [Classification Models Scoring](#classification-models-scoring)
    - [Test-Time Augment](#test-time-augment)
- [Results](#results)
- [Discussion and Conclusion](#discussion-and-conclusion)
- [Future Ideas](#future-ideas)
- [Further Reading](#further-reading)
- [References](#references)

## Dataset

We are using the xview2 [^xview2] challenge dataset, namely Xbd[^xbd], as the dataset for our project. This dataset contains pairs of
pre and post-disaster images from 19 natural disasters worldwide, including fires, hurricanes, floods, and
earthquakes. Each sample in the dataset consists of a pre-disaster image with its building annotations and a
post-disaster image with the same building annotations. However, in the post-disaster building annotations, each
building has a damage level of the following: *undamaged*, *damage*, *major damage*, *destroyed*, and *
unclassified*. The dataset consists of *train*, *tier3*, *test*, and *hold* subsets. Each subset has an *images* folder
containing pre and post-disaster images stored as 1024\*1024 PNGs and a folder named *labels* containing building
annotations and damage labels in JSON format. Some of the post-imagery is slightly shifted from their corresponding
pre-disaster image. Also, the dataset has different ground sample distances. We used the *train* and *tier3* subsets for
training, the *test* subset for validation, and the *hold* subset for testing. The dataset is highly unbalanced in
multiple aspects. The buildings with the *undamaged* label are far more than buildings with other damage types. The
number of images varies a lot between different disasters; the same is true for the number of building annotations in
each disaster.

<details>
<summary>
Folder Structure
</summary>

```
dataset
├── test
|   └── ... (similar to train)
├── hold
|   └── ... (similar to train)
├── tier3
|   └── ... (similar to train)
└── train
    ├── images
    │   ├── ...
    │   ├── {disaster}_{id}_post_disaster.png
    │   └── {disaster}_{id}_pre_disaster.png
    ├── labels
    │   ├── ...
    │   ├── {disaster}_{id}_post_disaster.json
    │   └── {disaster}_{id}_pre_disaster.json
    └── targets
        ├── ...
        ├── {disaster}_{id}_post_disaster_target.png
        └── {disaster}_{id}_pre_disaster_target.png
```

</details>

<details>
<summary>Example Usage</summary>

```python
from pathlib import Path
from metadamagenet.dataset import LocalizationDataset, ClassificationDataset

dataset = LocalizationDataset(Path('/path/to/dataset/train'))
dataset = ClassificationDataset([Path('/path/to/dataset/train'), Path('/path/to/dataset/tier3')])
```

</details>



![an example of data](./res/data.png)

## Problem Definition

We can convert these building annotations (polygons) to a binary mask. We can also convert the damage levels to values
1-4 and use them as the value for all the pixels in their corresponding building, forming a semantic segmentation mask.
Thus, we define the building localization task as predicting each pixel's value being zero or non-zero. We also define
the damage classification task as predicting the exact value of pixels within each building. We consider the label of an
unclassified building as undamaged, as it is the most common label by far in the dataset.

## Data Augmentations

<details>
<summary>
Example Usage
</summary>

```python
import torch
from metadamagenet.augment import Random, VFlip, Rotate90, Shift, RotateAndScale, BestCrop, OneOf, RGBShift, HSVShift,\
    Clahe, GaussianNoise, Blur, Saturation, Brightness, Contrast, ElasticTransform

transform = torch.nn.Sequential(
    Random(VFlip(), p=0.5),
    Random(Rotate90(), p=0.95),
    Random(Shift(y=(.2, .8), x=(.2, .8)), p=.1),
    Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.1),
    BestCrop(samples=5, dsize=(512, 512), size_range=(0.45, 0.55)),
    Random(RGBShift().only_on('img'), p=0.01),
    Random(HSVShift().only_on('img'), p=0.01),
    OneOf(
        (OneOf(
            (Clahe().only_on('img'), 0.01),
            (GaussianNoise().only_on('img'), 0.01),
            (Blur().only_on('img'), 0.01)), 0.01),
        (OneOf(
            (Saturation().only_on('img'), 0.01),
            (Brightness().only_on('img'), 0.01),
            (Contrast().only_on('img'), 0.01)), 0.01)
    ),
    Random(ElasticTransform(), p=0.001)
)

inputs = {
    'img': torch.rand(3, 3, 100, 100),
    'msk': torch.randint(low=0, high=2, size=(3, 100, 100))
}
outputs = transform(inputs)
```

</details>

Data Augmentation techniques help generate new valid samples from the dataset. Hence, they provide us with more data,
help the model train faster, and prevent overfitting. Data Augmentation is vastly used in training computer vision
tasks, from image classification to instance segmentation. In most cases, data augmentation is done randomly. This
randomness means that the augmentation is not done on some of the original samples and it has some random parameters. Most
libraries used for augmentation, like Open-CV [^open-cv], do not support image-batch transforms and only perform transforms
on the CPU. Kornia [^kornia] [^kornia-survey] is an open-source differentiable computer vision library for PyTorch[^pytorch]; it supports
image-batch transforms and performs these transforms on GPU. We used Kornia and added some parts to
it to suit our project requirements.

We created a version of each image transformation in order for it to support our needs. 
Its input is multiple batches of images, and
each batch has a name. 
For example, an input contains a batch of images and a batch of corresponding segmentation masks.
In some transformations like resize, the same parameters (in this case, scale) should be used for transforming both
the images and the segmentation masks. In some transformations, like channel shift, the transformation should not be done on the
segmentation masks. Another requirement is that the transformation parameters can differ for each image and its
corresponding mask in the batch.
Furthermore, a random augmentation should generate different transformation parameters for each image in the batch.
Moreover, it should be considered that the transformation does not apply to some images in the batch. Our version of each
transformation meets these requirements.

## Methodology

<details>
<summary>Example Usage</summary>

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

</details>

### General Architecture

As shown in the figure below, building-localization models consist of a feature extractor (a U-net [^unet] or a SegFormer [^segformer]) and a
classifier module [^first_place_solution].
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
In many samples, the post-disaster image has a minor shift compared to the pre-disaster image. However,
the segmentation masks are created based on the buildings' location in the pre-disaster image.
This shift is an issue the model has to overcome. In our models, feature-extracting weights are shared between the two images. 
This helps the model to detect the shift or nadir difference. For models that share a joint feature extractor (like SegFormerB0 Classifier and SegFormerB0 Localizer),
we can initialize the feature extractor module in the classification model with the localization model's feature
extractor.
Since we do not use the localization model directly for damage assessment, training the localization model can be
seen as a pre-training stage for the classification model.

![General Architecture](./res/arch.png)

### U-Models

Some models in this project use a U-net [^unet] module as the feature extractor and a superficial 2D Convolutional Layer as the
classifier.
We call them U-models. Their feature extractor module is a U-net [^unet] with five encoder and five decoder modules.
Encoder modules are usually a part of a general feature extractor like *Resnet-34* [^resnet].
In the forward pass of each image through each encoder module, the number of channels may or may not change.
Still, the height and width of the image are divided by two.
Usually, the five encoder modules combined include all layers of a general feature extractor model (like Resnet34 [^resnet])
except for the classification layer.
Each decoder module combines the output of the previous decoder module and the respective encoder module.
For example, encoder module 2 combines the output of decoder module 1 and encoder module 3.
They form a U-like structure, as shown in the figure below.

![Unet](./res/unet-architecture.png)

#### Decoder Modules

There are two variants of decoder modules:
The *Standard* decoder module and the *SCSE* [^SCSE] decoder module.
The *Standard* decoder module applies a 2D convolution and a *Relu* activation to the input from the previous decoder.
Then, it concatenates the result with the input from the respective encoder module and applies another 2D convolution, and *ReLU* activation.
*SCSE* decoder module works the same way, but in the last step,
it uses a "Concurrent Spatial and Channel Squeeze & Excitation" [^SCSE] module on the result.
This SCSE module is supposed to help the model focus on the image's more critical regions and channels.
Decoder modules in *xview2 first place solution* [^first_place_solution] don't use batch normalization between the convolution and the activation.
We added this layer to the decoder modules to prevent gradient exploding and to make these modules more stable.

![Decoder Modules](./res/decoder.png)

#### Backbone

We pick encoder modules of U-net modules from a general feature extractor model called *The Backbone Network*.
The choice of the backbone network is the most crucial point in the performance of a U-net model.
Plus, most of the parameters of a U-net model are of its backbone network.
Thus, the choice of the backbone network significantly impacts its size and performance.
*xview2 first place solution* [^first_place_solution] used *Resnet34* [^resnet], *Dual Path Network 92* [^dpn], *SeResnext50 (32x4d)* [^resnext], and *SeNet154* [^SeNet] as the backbone
network.
We used *EfficientNet B0* and *EfficientNet B4* [^efficientnet] (both standard and *Wide-SE* versions) as the backbone network, creating
new U-models called *Efficient-Unets*.
*EfficientNets* [^efficientnet] have shown excellent results on the ImageNet [^imagenet] dataset, so they are good feature extractors.
They are also relatively small in size. These two features make them perfect choices for a backbone network.

We listed all the used models and their attributes in the table below.

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
  <tr>
    <td>SegFormer</td>
    <td>segformer_512*512_ade</td>
    <td>3,714,401</td>
    <td colspan="2"></td>
  </tr>
</table>

### Meta-Learning

In meta-learning, a general problem, such as classifying different images (in the *ImageNet* dataset) or
different letters (in the *Omniglot* [^omniglot] dataset), is seen as a distribution of tasks. In this approach, tasks are generally
the same problem (like classifying letters) but vary in some parameters (like the script letters belong to).
We can take a similar approach to our problem. We can view building detection and damage level classification as the
general problem and take the disaster type (like a flood, hurricane, or wildfire) and the environment of the disaster
(like a desert, forest, or urban area) as the varying factors. In distance-learning methods, the distance function
returns a distance between the query sample and each class's sample. Then, the query sample is classified into the
class with the minimum distance. These methods are helpful when we have a high number of classes. However, in our case,
the number of classes is fixed. Thus, we used a model-agnostic approach. Model agnostic meta-learning [^maml] algorithms find a
set of parameters for the model that can be adapted to a new task by training with very few samples.
We used the MAML [^maml] algorithm and considered every different disaster a separate task.
Since the MAML algorithm consumes lots of memory, and the consumed memory is
relative to the model size, we have used models based on EfficientUnetB0 and
only trained it for the building localization task.

Since the MAML algorithm trains the model much slower than regular training,
and we had limited time to train our models, the results weren't satisfactory.
We trained EfficientUnetB0-Localizer with MAML with support shots equal to one or five
and query shots equal to two or ten. Other training hyperparameters
and evaluation results are available in the results section.
We utilized the *Higher* [^higher] library to implement the MAML algorithm.


<details>
<summary>
Example Usage
</summary>

```python
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from metadamagenet.dataset import discover_directory, group_by_disasters, MetaDataLoader, LocalizationDataset, ImageData
from metadamagenet.metrics import xview2
from metadamagenet.losses import BinaryDiceLoss
from metadamagenet.runner import MetaTrainer, MetaValidationInTrainingParams
from metadamagenet.models import BaseModel

dataset: list[ImageData] = discover_directory
tasks: list[tuple[str, list[ImageData]]] = group_by_disasters(dataset)

train = MetaDataLoader(LocalizationDataset, tasks[:-2], task_set_size=17, support_shots=4, query_shots=8, batch_size=1)
test = MetaDataLoader(LocalizationDataset, tasks[-2:], task_set_size=2, support_shots=4, query_shots=8, batch_size=1)

model: BaseModel
version: str
seed: int
meta_optimizer: Optimizer
inner_optimizer: Optimizer
lr_scheduler: MultiStepLR
MetaTrainer(
    model,
    version,
    seed,
    train,
    nn.Identity(),
    meta_optimizer,
    inner_optimizer,
    lr_scheduler,
    BinaryDiceLoss(),
    epochs=50,
    n_inner_iter=5,
    score=xview2.localization_score,
    validation_params=MetaValidationInTrainingParams(
        meta_dataloader=test,
        interval=1,
        transform=None,
    )
).run()
```

</details>

### Vision Transformer

In recent years, vision transformers [^vit] have achieved state-of-the-art results in many computer vision tasks, including
semantic segmentation. SegFormer [^segformer] is a model designed for efficient semantic segmentation, and it is based on vision
transformers. SegFormer is available in different sizes. We only used the smallest size, named SegFormerB0. The
SegFormer model consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head.
In contrast to U-nets, SegFormer models have constant input and output sizes. So, the inputs and outputs of the model should be
interpolated to the correct size. For the localization task, the input image goes through SegFormer, and its outputs go
through a SegFormer decode head.
However, for the classification task, pre and post-disaster go through the same Segformer model. Next, their outputs are
concatenated channel-wise and then go through a modified SegfFormer decode head. The modification is to double the
number of channels for the MLP modules. Of course, both outputs can be merged in successive layers, which decreases the
distance function complexity. These other versions of the modified decode head can be created and tested in the future.
Moreover, one can experiment with changing the size of the SegFormer input and SegFormer model size.

### Training Setup

We trained some models with multiple random seeds (multiple folds) to ensure they have low variance and consistent
scores. We trained Localization models only on pre-disaster images because post-disaster images added noise to the data; we used post-disaster images in sporadic cases as
additional augmentation. We initialized each classification model's feature extractor using weights from the
corresponding localization model and fold number. In training both classification and localization models, no weights
were frozen. Since the dataset is unbalanced, we use weighted losses with weights relative to the inverse of each class's
sample count. We applied morphological dilation with a 5*5 kernel to classification masks as an augmentation. Dilated
masks made predictions bolder. We also used PyTorch [^pytorch] amp for FP-16 [^fp16] training.

### Loss Functions

<details>
<summary>
Example Usage
</summary>

```python
from metadamagenet.losses import WeightedSum, BinaryDiceLoss, BinaryFocalLoss

WeightedSum(
    (BinaryDiceLoss(), 1.0),
    (BinaryFocalLoss(alpha=0.7, gamma=2., reduction='mean'), 6.0)
)
```

</details>

Both Building Localization and Damage Classification are semantic segmentation tasks.
Because, in both problems, the model's purpose is classification at pixel level.
We have used a combination of multiple segmentation losses for all models.
In [^segmentation-losses],
you can find a comprehensive comparison between popular loss functions for semantic segmentation.

Focal and Dice Loss are loss functions used in the training localization models.
For Classification models, we tried channel-wise-weighted versions of Focal, Dice, and Cross-entropy Loss.

**Focal Loss**[^focal-loss]

$$
FL(p_t) = -\alpha_t(1- p_t)\gamma log(p_t).
$$

Focal Loss's usage is to make the model focus on hard-to-classify examples by increasing their loss value. We used it
because the target distribution was highly skewed. In the building localization task, the number of pixels containing
buildings was far less than the background pixels. In the damage classification task, undamaged building samples
formed most of the total samples, too.

**Dice Loss**[^dice-loss]

$$
Dice\space Loss(p,t) = 1 - dice(p,t)
$$

Where $dice$, $p$ and $t$ stand for *dice coefficient*, *predictions* and *target values* respectively.

$$
dice(A,B) = 2\frac{ A\cap B}{A + B}
$$

Dice loss is calculated globally over each mini-batch. For multiclass cases, the loss value of each class (channel) is
calculated individually, and their average is used as the final loss. Two activation functions can be applied to model
outputs before calculating dice loss: *sigmoid* and *softmax*. Softmax makes the denominator of the final loss function
constant and thus has less effect on the model's training, though it makes better sense.


**Cross Entropy Loss**[^cross-entropy]

$$
-\sum_{c=1}^My_{o,c}\log(p_{o,c})
$$

Since we used sigmoid-dice-loss for multiclass damage classification, cross-entropy loss helped the model assign only
one class to each pixel. It solely is a good loss function for semantic segmentation tasks.

## Evaluation

<details>
<summary>
Example Usage
</summary>

```python
from torch import Tensor
from metadamagenet.metrics import DamageLocalizationMetric, DamageClassificationMetric

evaluator = 0.2 * DamageLocalizationMetric() + 0.8 * DamageClassificationMetric()

preds: Tensor
targets: Tensor
score = evaluator(preds, targets)
```

</details>

One of the most popular evaluation metrics for classifiers is the F1-score because it accounts for precision and recall
simultaneously. The macro version of the F1-score is a good evaluation measure for imbalanced datasets. The
[xview2-scoring](https://github.com/DIUx-xView/xView2_scoring) repository describes what variation of F1-score to use for this problem's scoring. We adapted their
evaluation metrics. However, we implemented these metrics as a metric in the Torchmetrics [^torchmetrics] library. It performs
better than computing metrics in NumPy [^numpy] and provides an easy-to-use API. The dice score is a set similarity measure that equals the F1-score.

$$
Dice(P,Q) = 2. \frac{P \cap Q}{P+Q}
$$

$$
F1(P,Q) = \frac{2TP}{2TP + FP + FN}
$$

### Localization Models Scoring

The localization score is defined as a globally calculated binary f1-score. Sample-wise calculation means calculating the
score on each sample (image) and then averaging sample scores to get the final score. In global calculation, we use the sum
of true positives, true negatives, false positives, and false negatives across all samples to calculate the metric.

The localization score is a binary f1-score, which means class zero (no-building/background) is considered negative, and
class one (building) is considered positive. Since we only care about detecting buildings from the background,
micro-average is applied too.

### Classification Models Scoring

The classification score consists of a weighted sum of 2 scores: the localization score and the damage classification
score. Classification models a label of zero to four for each pixel, indicating no-building, no damage, minor damage,
major damage, and destroyed, respectively. Since one to four label values show that a specific pixel belongs to a building, we
calculate the localization score after converting all values above zero to one. This score determines how good the model
is at segmenting buildings. We define the damage classification score as the harmonic mean of the globally computed
f1-score for each class from one to four. We calculate the f1-score of each class separately, then use their harmonic
mean to give each damage level equal importance. Here, we prefer the harmonic mean to the arithmetic mean because
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

### Test-Time Augment

![Test-Time Augment](./res/TTA.png)

While validating a model, we give each piece (or mini-batch) of data to the model and compute a score by comparing the
model output and the correct labels. Test-time augment is a technique to enhance the accuracy of the predictions by
eliminating the model's bias. For each sample, we use reversible augmentations to generate multiple "transformed
samples". The predicted label for the original sample computes as the average of the predicted labels for the "
transformed samples". For example, we generate the transformed samples by rotating the original image by 0, 90, 180, and
270 degrees clockwise. Then, we get the model predictions for these transformed samples. Afterward, we rotate the
predicted masks 0, 90, 180, and 270 degrees counterclockwise and average them. Their average counts as the model's
prediction for the original sample. Using this technique, we eliminate the model's bias of rotation. By reversible
augmentation, we mean that no information should be lost during the process of generating "transformed samples" and
aggregating their results. For example, in the case of semantic segmentation, shifting an image does not count as a
reversible augmentation because it loses some part of the image. However, this technique usually does not improve the
performance of well-trained models much. Because their bias of a simple thing like rotation is tiny. The same was
true for our models when we used flipping and 90-degree rotation as test-time augmentation.

<details>
<summary>
Example Usage
</summary>

```python
from metadamagenet.models import FourFlips, FourRotations, BaseModel

model: BaseModel
model_using_test_time_augment = FourRotations(model)
```

</details>

## Results

Using pre-trained feature extractors from localization models allowed
classification models to train much faster and have higher scores.
Using dilated masks improved accuracy around borders and helped with shifts and
different nadirs. The model's classifier module determines each pixel's value
based on a distance function between the extracted features from the
pre- and post-disaster images.
In U-models, the classifier module is a 2D convolution, but in SegFormer models,
it is a SegFormer decoder head. Hence, U-models learn a much simpler distance
function than SegFormer models; the simplicity of the distance function helps them not to overfit but also prevents them from learning some sophisticated patterns. In the end, SegFormer models train much faster before overfitting on the training data, but U-models slowly reach almost the same score. EfficientUnet localization models have shown that they train better without using focal loss. Softmax dice loss does not perform well in the damage classification model's training. A combination of sigmoid dice loss for each class (channel), and cross-entropy loss gives the best results in the training of a classification model. The effect of SCSE in decoder modules and Wide-SE in Encoder Modules of a U-net is very limited; these variations of EfficientUnets performed almost the same as the standard version.

complete results are available at [results.md](./results.md)

## Discussion and Conclusion
Detecting buildings and their damage level by artificial intelligence can improve rescue operations' speed and efficiency after natural disasters. Solving this problem can identify the area of damage on a large scale and prioritize the areas that have been the most affected and damaged by a disaster in rescue operations. We tested different decoders in the U-net modules and utilized different variations of efficient-net as the backbone in our model. Additionally, we fine-tuned SegFormer for our specific task. The result was models with fewer parameters (approximately three million) that performed much better than the previous models (damage classification score=0.77). Due to the fewer parameters, these models have a shorter training and inference time. Therefore, they can be trained and used faster and easily fine-tuned for new and different natural disasters. Considering damage classification and building localization in each natural disaster as a separate task, we utilized MAML and trained models that can be adapted to a new natural disaster using only a few brand-new samples. These models do not have satisfactory performance, but we hope to build better models of this type in the future.


## Future Ideas

The decoder’s number of channels can be looked at as a hyper-parameter
which can be changed and tuned.
Additionally, we can analyze the effect of the size of the backbone in
the efficient U-net by trying Efficientnet b5 or b7 as the backbone.
The layer in which the embedding of the pre- and post-disaster images
get concatenated dictates the complexity of the distance function in the classifier.
This effect can also be tested and analyzed.
*Log-cosh-dice* and *Focal-Travesky* are two loss functions that
have the best performance in the training of segmentation models [^segmentation-losses].
We can also try training our models with these two loss functions.
But in this case, we have to make sure to modify them, so we can assign weights to classes.
The low performance of the meta learning model may
not be only due to the small number of training epochs or the small number of shots.
We can try using first-order MAML like Reptile [^reptile] instead of the original MAML algorithm in the model.
These algorithms use less memory, thus, we can test the effects of other factors and hyperparameters faster.
Previous research in the realm of meta-learning for semantic segmentation may also help us train a better model for our specific problem. [^meta-seg] [^meta-seg-init].


## Further Reading

- :book: [Higher Repository](https://github.com/facebookresearch/higher)
- :link: [Squeeze and Excitation Networks Explained with PyTorch Implementation](https://amaarora.github.io/2020/07/24/SeNet.html)
- :link: [Xview2](https://github.com/ethanweber/xview2)
- :link: [Workshop on Meta-Learning (MetaLearn 2022)](https://meta-learn.github.io/2022/)
- :link: [Meta-Learning with Implicit Gradients](https://sites.google.com/view/imaml)
- :link: [Learning About Algorithms That Learn to Learn](https://towardsdatascience.com/learning-about-algorithms-that-learn-to-learn-9022f2fa3dd5)
- :link: [A Search for Efficient Meta-Learning: MAMLs, Reptiles, and Related Species](https://towardsdatascience.com/a-search-for-efficient-meta-learning-mamls-reptiles-and-related-species-e47b8fc454f2)
- :link: [Learn To Learn: A blog post from an author of MAML](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)
- :link: [Comparing F1/Dice score and IoU](https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou)
- :link: [Xview2 Baseline Repository](https://github.com/DIUx-xView/xView2_baseline)
- :link: [Concurrent Spatial and Channel Squeeze & Excitation (scSE) Nets](https://blog.paperspace.com/scse-nets/)
- :link: [Channel Attention and Squeeze-and-Excitation Networks (SENet)](https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/)
- :link: [Introduction to ResNets](https://towardsdatascience.com/introduction-to-resnets-c0a830a288a4)
- :link: [Understand Deep Residual Networks — a simple, modular learning framework that has redefined state-of-the-art](https://medium.com/@waya.ai/deep-residual-learning-9610bb62c355)
- :link: [Review: ResNeXt — 1st Runner Up in ILSVRC 2016 (Image Classification)](https://towardsdatascience.com/review-resnext-1st-runner-up-of-ilsvrc-2016-image-classification-15d7f17b42ac)
- :link: [A Review of Popular Deep Learning Architectures: DenseNet, ResNeXt, MnasNet, and ShuffleNet v2](https://blog.paperspace.com/popular-deep-learning-architectures-densenet-mnasnet-shufflenet/)

- :link: [Cadene/Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch)

## References

[^xbd]: :page_facing_up: [xBD: A Dataset for Assessing Building Damage](https://arxiv.org/abs/1911.09296)

[^xview2]: :link: Competition and Dataset: [Xview2 org.](https://www.xview2.org)

[^first_place_solution]: :link: [Xview2 First Place Solution](https://github.com/vdurnov/xview2_1st_place_solution)

[^unet]: :page_facing_up: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[^segformer]: :page_facing_up: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

[^higher]: :page_facing_up: [Generalized Inner Loop Meta-Learning](https://arxiv.org/abs/1910.01727)

[^maml]: :page_facing_up: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

[^multi-temporal-fusion]: :page_facing_up: [Building Disaster Damage Assessment with Multi-Temporal Fusion](https://arxiv.org/abs/2004.05525)

[^SCSE]: :page_facing_up: [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)

[^SeNet]: :page_facing_up: [Recalibrating Fully Convolutional Networks with Spatial and Channel 'Squeeze & Excitation' Blocks](https://arxiv.org/abs/1808.08127)

[^fp16]: :page_facing_up: [AMPT-GA: Automatic Mixed Precision Floating Point Tuning for GPU Applications](https://engineering.purdue.edu/dcsl/publications/papers/2019/gpu-fp-tuning_ics19_submitted.pdf)

[^dpn]: :page_facing_up: [Dual Path Networks](https://arxiv.org/abs/1707.01629)

[^meta-seg]: :page_facing_up: [Meta-seg: A survey of meta-learning for image segmentation](https://www.sciencedirect.com/science/article/abs/pii/S003132032200067X)

[^meta-seg-init]: :page_facing_up: [Meta-Learning Initializations for Image Segmentation](https://meta-learn.github.io/2020/papers/44_paper.pdf)

[^focal-loss]: :page_facing_up: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

[^reptile]: :page_facing_up: [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)

[^imaml]: :page_facing_up: [Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630)

[^resnet]: :page_facing_up: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[^resnext]: :page_facing_up: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

[^efficientnet]: :page_facing_up: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

[^kornia]: :page_facing_up: [Kornia: an Open Source Differentiable Computer Vision Library for PyTorch](https://arxiv.org/abs/1910.02190)

[^kornia-survey]: :page_facing_up: [A survey on Kornia: an Open Source Differentiable Computer Vision Library for PyTorch](https://arxiv.org/abs/2009.10521)


<!--https://github.com/kornia/kornia/blob/master/CITATION.md -->

[^open-cv]: :page_facing_up: [open cv](https://github.com/opencv/opencv)

<!-- 
@article{opencv_library,
    author = {Bradski, G.},
    citeulike-article-id = {2236121},
    journal = {Dr. Dobb's Journal of Software Tools},
    keywords = {bibtex-import},
    posted-at = {2008-01-15 19:21:54},
    priority = {4},
    title = {{The OpenCV Library}},
    year = {2000}
}
-->

[^dice-loss]: :page_facing_up: [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237)

[^segmentation-losses]: :page_facing_up: [A survey of loss functions for semantic segmentation](https://arxiv.org/abs/2006.14822)

[^cross-entropy]: :page_facing_up: [Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels](https://arxiv.org/abs/1805.07836)

[^pytorch]: :page_facing_up: [Automatic differentiation in PyTorch](https://openreview.net/pdf?id=BJJsrmfCZ)

[^imagenet]: :page_facing_up: [ImageNet: A large-scale hierarchical image database](https://ieeexplore.ieee.org/document/5206848)

[^omniglot]: :page_facing_up: [The Omniglot challenge: a 3-year progress report](https://arxiv.org/abs/1902.03477v2)

[^vit]: :page_facing_up: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)

[^torchmetrics]: :link: [Machine learning metrics for distributed, scalable PyTorch applications.](https://github.com/Lightning-AI/metrics)

[^numpy]: :page_facing_up: [Array programming with NumPy](https://numpy.org/citing-numpy/)
