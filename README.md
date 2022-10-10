# MetaDamageNet
#### Using Deep Learning To Identify And Classify Damage In Aerial Imagery

Some ideas from this project is borrowed from [xview2 first place solution](https://github.com/vdurnov/xview2_1st_place_solution) 
repository. I used that repository as a baseline.
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

# Architecture

## Model

![Model](./res/model.png)

## Unet

![Unet](./res/unet-architecture.png)

### Module

![Decoder Modules](./res/decoder.png)



# Data Cleaning Techniques

Dataset for this competition well prepared and I have not found any problems with it.
Training masks generated using json files, "un-classified" type treated as "no-damage" (create_masks.py). "masks"
folders will be created in "train" and "tier3" folders.

The problem with different nadirs and small shifts between "pre" and "post" images solved on models level:

- Frist, localization models trained using only "pre" images to ignore this additional noise from "post" images. Simple
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

All models trained with Train/Validation random split 90%/10% with fixed seeds (3 folds). Only checkpoints from epoches
with best validation score used.

For localization models 4 different pretrained encoders used:
from torchvision.models:

- ResNet34
  from https://github.com/Cadene/pretrained-models.pytorch:
- se_resnext50_32x4d
- SeNet154
- Dpn92

Localization models trained on "pre" images, "post" images used in very rare cases as additional augmentation.

Localization training parameters:
Loss: Dice + Focal
Validation metric: Dice
Optimizer: AdamW

Classification models initilized using weights from corresponding localization model and fold number. They are Siamese
Neural Networks with whole localization model shared between "pre" and "post" input images. Features from last Decoder
layer combined together for classification. Pretrained weights are not frozen.
Using pretrained weights from localization models allowed to train classification models much faster and to have better
accuracy. Features from "pre" and "post" images connected at the very end of the Decoder in bottleneck part, this
helping not to overfit and get better generalizing model.

Classification training parameters:
Loss: Dice + Focal + CrossEntropyLoss. Larger coefficient for CrossEntropyLoss and 2-4 damage classes.
Validation metric: competition metric
Optimizer: AdamW
Sampling: classes 2-4 sampled 2 times to give them more attention.

Almost all checkpoints finally finetuned on full train data for few epoches using low learning rate and less
augmentations.

Predictions averaged with equal coefficients for both localization and classification models separately.

Different thresholds for localization used for damaged and undamaged classes (lower for damaged).

## Augmentations

## Test-Time Augment

## Unet Models

## MetaLearning

# Conclusion and Acknowledgments

Thank you to xView2 team for creating and releasing this amazing dataset and opportunity to invent a solution that can
help to response to the global natural disasters faster. I really hope it will be usefull and the idea will be improved
further.

# Evaluation Results

on dataset `test` (used for validation):

<table>
<thead>
  <td colspan="1">model</td>
  <td colspan="1">version</td>
  <td colspan="8">Localization</td>
  <td colspan="8">Classification</td>
</thead>
<thead>
  <td colspan="2">seed</td>
  <td colspan="2">0</td>
  <td colspan="2">1</td>
  <td colspan="2">2</td>
  <td colspan="2">mean</td>
  <td colspan="2">0</td>
  <td colspan="2">1</td>
  <td colspan="2">2</td>
  <td colspan="2">mean</td>
</thead>
<thead>
  <td colspan="2"> TTA </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="2"> - </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="2"> - </td>
</thead>
<tr>
  <td>Resnet34Unet</td>
  <td>1</td>
  <td>0.6555</td>
  <td>0.6593</td>

  <td>0.6675</td>
  <td>0.6742</td>

  <td>0.6820</td>
  <td>0.6837</td>

  <td colspan="2">0.6731</td>

  <td>0.3608</td>
  <td>0.3203</td>

  <td>0.4566</td>
  <td>0.4603</td>

  <td>0.4284</td>
  <td>0.4310</td>

  <td colspan="2">0.4164</td>
</tr>
<tr>
  <td>SeResnext50Unet</td>
  <td>tuned</td>
  <td>0.6943</td>
  <td>0.6917</td>

  <td>0.6922</td>
  <td>0.6952</td>

  <td>0.7000</td>
  <td>0.7030</td>

  <td colspan="2">0.7017</td>

  <td>0.6751</td>
  <td>0.6712</td>

  <td>0.6428</td>
  <td>0.6419</td>

  <td>0.6601</td>
  <td>0.6703</td>

  <td colspan="2">0.6687</td>
</tr>
<tr>
  <td>Dpn92Unet</td>
  <td>tuned</td>
  <td>0.6774</td>
  <td>0.6825</td>

  <td>0.6338</td>
  <td>0.6313</td>

  <td>0.6654</td>
  <td>0.6720</td>

  <td colspan="2">0.6644</td>

  <td>0.6747</td>
  <td>0.6818</td>

  <td>0.6292</td>
  <td>0.6264</td>

  <td>0.6397</td>
  <td>0.6485</td>

  <td colspan="2">0.6689</td>
</tr>
<tr>
  <td>SeNet154Unet</td>
  <td>1</td>

  <td>0.7246</td>
  <td>0.7289</td>

  <td>0.7107</td>
  <td>0.7168</td>

  <td>0.7221</td>
  <td>0.7244</td>

  <td colspan="2">0.7282</td>

  <td>0.6963</td>
  <td>0.7012</td>

  <td>0.6153</td>
  <td>0.6418</td>

  <td>0.6862</td>
  <td>0.6838</td>

  <td colspan="2">0.6982</td>
</tr>
<tr>
  <td>EfficientUnetB0</td>
  <td>00</td>
  <td>0.75896</td>
</tr>
<tr>
  <td>EfficientUnetWideSEB0</td>
  <td>00</td>
  <td>0.75884</td>
</tr>
<tr>
  <td>EfficientUnetB0SCSE</td>
  <td>00</td>
  <td>0.75886</td>
</tr>
<tr>
  <td>EfficientUnetB4</td>
  <td>00</td>
  <td>0.76844</td>
</tr>
<tr>
  <td>EfficientUnetB4SCSE</td>
  <td>00</td>
  <td>0.76553</td>
</tr>
</table>

on dataset `hold` (used for testing):
<table>
<thead>
  <td colspan="1">model</td>
  <td colspan="1">version</td>
  <td colspan="8">Localization</td>
  <td colspan="8">Classification</td>
</thead>
<thead>
  <td colspan="2">seed</td>
  <td colspan="2">0</td>
  <td colspan="2">1</td>
  <td colspan="2">2</td>
  <td colspan="2">mean</td>
  <td colspan="2">0</td>
  <td colspan="2">1</td>
  <td colspan="2">2</td>
  <td colspan="2">mean</td>
</thead>
<thead>
  <td colspan="2"> TTA </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="2"> - </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="1"> - </td>
  <td colspan="1"> + </td>

  <td colspan="2"> - </td>
</thead>
<tr>
  <td>Resnet34Unet</td>
  <td>1</td>

  <td>0.6609</td>
  <td>0.6667</td>

  <td>0.6685</td>
  <td>0.6779</td>

  <td>0.6842</td>
  <td>0.6882</td>

  <td colspan="2">0.7085</td>
</tr>
<tr>
  <td>SeResnext50Unet</td>
  <td>tuned</td>

  <td>0.6953</td>
  <td>0.6964</td>

  <td>0.7020</td>
  <td>0.7115</td>

  <td>0.7049</td>
  <td>0.7095</td>

  <td colspan="2">0.7093</td>
</tr>
<tr>
  <td>Dpn92Unet</td>
  <td>tuned</td>

  <td>0.6812</td>
  <td>0.6832</td>

  <td>0.6294</td>
  <td>0.6317</td>

  <td>0.6688</td>
  <td>0.6732</td>

  <td colspan="2">0.6646</td>
</tr>
<tr>
  <td>SeNet154Unet</td>
  <td>1</td>

  <td>0.7340</td>
  <td>0.7394</td>

  <td>0.7244</td>
  <td>0.7306</td>

  <td>0.7347</td>
  <td>0.7392</td>

  <td colspan="2">0.7402</td>
</tr>
</table>


Unet Models:
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
    <td>Normal</td>
  </tr>
  <tr>
    <td>SeResnext50Unet</td>
    <td>se_resnext50_32x4d</td>
    <td>34,559,728</td>
    <td>No</td>
    <td>Normal</td>
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
    <td>Normal</td>
  </tr>
  <tr>
    <td>EfficientUnetB0</td>
    <td rowspan="2">efficientnet_b0</td>
    <td>6,884,876</td>
    <td>Yes</td>
    <td>Normal</td>
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
    <td>Normal</td>
  </tr>
  <tr>
    <td>EfficientUnetB4</td>
    <td rowspan="2">efficientnet_b0</td>
    <td>20,573,144</td>
    <td>Yes</td>
    <td>Normal</td>
  </tr>
  <tr>
    <td>EfficientUnetB4SCSE</td>
    <td>20,592,128</td>
    <td>Yes</td>
    <td>SCSE- no concat</td>
  </tr>
</table>

## References
- Competition and Dataset: [Xview2 org.](https://www.xview2.org)
- [Xview2 First Place Solution](https://github.com/vdurnov/xview2_1st_place_solution)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Cadene/Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch)
