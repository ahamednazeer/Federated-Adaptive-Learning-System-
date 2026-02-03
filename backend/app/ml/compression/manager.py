"""
Model Compression Module
Handles dynamic resource-aware model shrinking using pruning and quantization
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
from typing import Dict, Optional, Tuple

class ModelCompressor:
    """
    Manages model compression techniques for edge deployment
    """
    
    def __init__(self):
        # Set quantization engine
        # qnnpack is generally supported on ARM (M1/M2) and x86
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        else:
            print("Warning: No supported quantization engine found.")

    def prune_model(self, model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        Apply L1 unstructured pruning to Linear and Conv2d layers
        
        Args:
            model: PyTorch model
            amount: Fraction of weights to prune (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        # Apply pruning to all suitable layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Make pruning permanent
                prune.remove(module, 'weight')
        
        return model

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization (weights to int8)
        
        Args:
            model: PyTorch model
            
        Returns:
            Quantized model
        """
        # Dynamic quantization for Linear and LSTM layers
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.LSTM, nn.GRU}, 
            dtype=torch.qint8
        )
        
        return quantized_model

    def optimize_for_device(self, model: nn.Module, device_constraints: Dict) -> nn.Module:
        """
        Select best optimization strategy based on device constraints
        
        Args:
            model: PyTorch model
            device_constraints: Dict with 'memory_limit_mb', 'compute_capability'
            
        Returns:
            Optimized model
        """
        memory_limit = device_constraints.get('memory_limit_mb', 1000)
        
        # Estimate model size (very rough approximation)
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        total_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        print(f"Current model size: {total_size_mb:.2f} MB")
        
        # Strategy selection
        optimized_model = model
        
        # If model is too big, apply pruning first
        if total_size_mb > memory_limit * 0.8:
            print(f"Model exceeds 80% of memory limit ({memory_limit} MB). Pruning...")
            optimized_model = self.prune_model(optimized_model, amount=0.4)
            
        # Always apply quantization for edge devices unless high compute capability
        if device_constraints.get('compute_capability', 'low') in ['low', 'medium']:
            print("Applying quantization for edge deployment...")
            optimized_model = self.quantize_model(optimized_model)
            
        return optimized_model

    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        torch.save(model.state_dict(), "temp.p")
        size_mb = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size_mb

def create_compressor() -> ModelCompressor:
    return ModelCompressor()
