import os
import io
import torch
import torch.nn as nn
from typing import Union

def get_model_size_mb(model: nn.Module) -> float:
    """
    Measure the memory size of the PyTorch model in megabytes.
    Handles standard parameters and dynamically quantized packed weights.
    """
    # Write weights to an in-memory buffer to get the exact serialized size
    buffer = io.BytesIO()
    try:
        # Use torch.save on state_dict, which handles serialized size well
        torch.save(model.state_dict(), buffer)
        return len(buffer.getvalue()) / (1024 * 1024)
    except Exception:
        # Fallback to parameter size summation if serialization fails
        total_bytes = 0
        # Standard parameters
        for p in model.parameters():
            total_bytes += p.nelement() * p.element_size()
        # Buffers (like masks)
        for b in model.buffers():
            total_bytes += b.nelement() * b.element_size()
        # Quantized modules pack weights in custom attributes
        for m in model.modules():
            if hasattr(m, '_packed_params') and hasattr(m._packed_params, '_packed_weight'):
                # Handle quantized linear layer weights
                weight = m._packed_params._packed_weight
                if hasattr(weight, 'dequantize'):
                    total_bytes += weight.nelement() * 1 # 1 byte for INT8
                else:
                    total_bytes += weight.nelement() * 4 # fallback
        return total_bytes / (1024 * 1024)

def quantize_model(model: nn.Module) -> nn.Module:
    """
    Apply post-training dynamic quantization to the model's Linear layers,
    mapping FP32 weights to INT8.
    
    Note: PyTorch's dynamic quantization is highly optimized for CPUs.
    Quantized models must run on CPU; they will error if run on GPU.
    """
    # Move model to CPU first, since dynamic quantization happens on CPU
    cpu_model = model.to("cpu")
    
    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        cpu_model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )
    
    print("Applied post-training dynamic INT8 quantization to Linear layers.")
    return quantized_model
