from typing import Dict, Tuple
import copy

import torch
import torch.nn.functional as tf
from torch import Tensor
from transformers import SegformerDecodeHead, SegformerModel, SegformerConfig, SegformerForSemanticSegmentation

from .base import BaseModel


class SegFormerLocalizer(BaseModel):
    def __init__(self):
        super().__init__()
        self.config: SegformerConfig = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.config.num_labels = 1
        self.segformer: SegformerModel = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.decode_head: SegformerDecodeHead = SegformerDecodeHead(self.config)
        self.segformer_input_size = (512, 512)

    @classmethod
    def name(cls) -> str:
        return "SegFormerLocalizer"

    def activate(self, outputs: Tensor) -> Tensor:
        return torch.sigmoid(outputs)

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data: {'img': torch.Tensor of shape (N,3,H,H),
                      'msk': torch.Tensor of shape (N,1,H,H)}
        :return: (inputs: torch.FloatTensor of shape (N,3,H,H),targets: torch.LongTensor of shape (N,H,W)
        """
        return (data['img'] * 2 - 1), data['msk'].long().squeeze(1)
        # todo: change normalization according to feature extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return_dict: bool = True
        output_hidden_states: bool = False
        output_size: Tuple[int, int] = x.shape[-2:]

        pixel_values = tf.interpolate(x, size=self.segformer_input_size, mode='bilinear', align_corners=False)

        outputs = self.segformer(
            pixel_values,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        logits = self.decode_head(outputs.hidden_states)
        upsampled_logits = tf.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        return upsampled_logits


class SegFormerClassifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.config: SegformerConfig = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.segformer: SegformerModel = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.config.num_labels = 5
        self.config.hidden_sizes = [hidden_size * 2 for hidden_size in self.config.hidden_sizes]
        self.decode_head: SegformerDecodeHead = SegformerDecodeHead(self.config)
        self.segformer_input_size = (512, 512)

    @classmethod
    def name(cls) -> str:
        return "SegFormerClassifier"

    def activate(self, outputs: Tensor) -> Tensor:
        return torch.softmax(outputs, dim=1)

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data: {'img_pre': torch.Tensor of shape (N,3,H,H),
                      'img_post': torch.Tensor of shape (N,3,H,H),
                      'msk':torch.Tensor of shape (N,1,H,H)}
        :return: (torch.FloatTensor of shape (N,6,H,H), torch.LongTensor of shape (N,H,W)
        """
        return (torch.cat((data['img_pre'] * 2 - 1, data['img_post'] * 2 - 1), dim=1),
                (data['msk'] * 4).long().squeeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return_dict: bool = True
        output_hidden_states: bool = False
        output_size: Tuple[int, int] = x.shape[-2:]

        pixel_values = tf.interpolate(x, size=self.segformer_input_size, mode='bilinear', align_corners=False)

        pre_outputs = self.segformer(
            pixel_values[:, :3, :, :],
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        post_outputs = self.segformer(
            pixel_values[:, 3:, :, :],
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        concatenated_outputs = [torch.cat([a, b], dim=1)
                                for a, b in zip(pre_outputs.hidden_states, post_outputs.hidden_states)]
        print([out.shape for out in concatenated_outputs])
        logits = self.decode_head(concatenated_outputs)
        upsampled_logits = tf.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        return upsampled_logits
