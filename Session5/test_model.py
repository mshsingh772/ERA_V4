import torch
import torch.nn as nn
from src.model import MNISTModel  # adjust based on your model import

def test_parameter_count():
    """Test if model has less than 20k parameters"""
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    assert total_params < 20000, f"Model has {total_params} parameters, expected < 20000"

def test_batch_norm_exists():
    """Test if model uses Batch Normalization"""
    model = MNISTModel()
    has_batch_norm = any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) 
                        for module in model.modules())
    print("\nBatch Normalization found:", has_batch_norm)
    assert has_batch_norm, "Model does not use Batch Normalization"

def test_dropout_exists():
    """Test if model uses Dropout"""
    model = MNISTModel()
    has_dropout = any(isinstance(module, nn.Dropout2d) 
                     for module in model.modules())
    print("\nDropout found:", has_dropout)
    assert has_dropout, "Model does not use Dropout"

def test_fc_or_gap_exists():
    """Test if model uses either Fully Connected Layer or Global Average Pooling"""
    model = MNISTModel()
    has_fc = any(isinstance(module, nn.Linear) 
                 for module in model.modules())
    has_gap = any(isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)) 
                  for module in model.modules())
    print(f"\nFC layer found: {has_fc}")
    print(f"GAP layer found: {has_gap}")
    assert has_fc or has_gap, "Model must use either Fully Connected Layer or Global Average Pooling" 