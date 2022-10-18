from typing import Dict, Tuple

import torch
import torch.nn.functional as tf
from torch import Tensor
from transformers import SegformerDecodeHead, SegformerModel, SegformerConfig, SegformerForSemanticSegmentation

from .base import BaseModel


class SegFormerLocalizer(BaseModel):
    def __init__(self):
        super().__init__()
        self.config: SegformerConfig = SegformerForSemanticSegmentation. \
            from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
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
