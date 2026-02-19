"""Provide IterDataPipe at torchdata.datapipes.iter by importing the
implementation from the PyTorch datapipe internals.
"""
from torch.utils.data.datapipes.datapipe import IterDataPipe

__all__ = ["IterDataPipe"]
