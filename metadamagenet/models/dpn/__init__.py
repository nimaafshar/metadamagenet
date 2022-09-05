""" PyTorch implementation of DualPathNetworks
Ported to PyTorch by [Ross Wightman](https://github.com/rwightman/pytorch-dpn-pretrained)

Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.

This implementation is compatible with the pretrained weights
from cypw's MXNet implementation.
"""
from .factory import dpn92
from .model import DPN
