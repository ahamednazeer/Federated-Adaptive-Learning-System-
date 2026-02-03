"""
Verification Script for New ML Features
Tests Model Shrinking, Continual Learning, and Adaptive Personalization
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# from app.main import app # Skipping main import to avoid overhead, testing ML modules directly

from app.ml.compression import ModelCompressor
from app.ml.federated.continual import EWC
from app.ml.federated.server import FederatedClient

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def test_compression():
    print("\n--- Testing Model Compression ---")
    model = SimpleModel()
    compressor = ModelCompressor()
    
    print("Original model created.")
    
    # Test Pruning
    pruned_model = compressor.prune_model(model, amount=0.5)
    print("Pruning applied. Verifying sparsity...")
    zero_weights = torch.sum(pruned_model.fc1.weight == 0).item()
    total_weights = pruned_model.fc1.weight.nelement()
    print(f"FC1 Sparsity: {zero_weights}/{total_weights} ({zero_weights/total_weights:.2f})")
    assert zero_weights > 0, "Pruning failed"
    
    # Test Quantization
    quantized_model = compressor.quantize_model(model)
    print("Quantization applied (dynamic).")
    # Note: Dynamic quantization modules change structure, checking if Linear became dynamic
    print(f"FC1 type: {type(quantized_model.fc1)}")
    
    print("✓ Compression Test Passed")

def test_continual_learning():
    print("\n--- Testing Continual Learning (EWC) ---")
    model = SimpleModel()
    ewc = EWC(model, fisher_multiplier=1000)
    
    # Create fake dataloader
    inputs = torch.randn(10, 10)
    labels = torch.randint(0, 2, (10,))
    dataloader = [(inputs, labels)]
    
    # Register task
    print("Registering task (computing Fisher matrix)...")
    ewc.register_task(dataloader, num_batches=1)
    
    assert len(ewc.fisher_matrix) > 0, "Fisher matrix empty"
    print(f"Fisher matrix computed for {len(ewc.fisher_matrix)} params.")
    
    # Debug: Print Fisher stats
    total_fisher = 0
    for name, f in ewc.fisher_matrix.items():
        s = f.sum().item()
        total_fisher += s
        # print(f"  {name}: {s}")
    print(f"Total Fisher Mass: {total_fisher}")

    # Calculate penalty
    loss = ewc.penalty_loss(model)
    print(f"EWC Penalty (should be 0 for same model): {loss.item()}")
    
    # Modify model slightly (make sure we modify a part with non-zero fisher)
    with torch.no_grad():
        # Find a param with non-zero fisher to modify
        for name, f in ewc.fisher_matrix.items():
            if f.sum().item() > 0:
                print(f"Modifying {name} which has fisher mass {f.sum().item()}")
                # Get the module and attribute
                module_name, attr_name = name.rsplit('.', 1)
                module = dict(model.named_modules())[module_name]
                getattr(module, attr_name).data += 0.5
                break
    
    loss_modified = ewc.penalty_loss(model)
    print(f"EWC Penalty after modification: {loss_modified.item()}")
    assert loss_modified > 0, f"EWC penalty should be positive after modification (Fisher mass: {total_fisher})"
    
    print("✓ Continual Learning Test Passed")

def test_adaptive_personalization():
    print("\n--- Testing Adaptive Personalization ---")
    model = SimpleModel()
    client = FederatedClient("test_client", model, "test_dataset")
    
    # Check requires_grad initial state
    print("Initial grad state:", model.fc1.weight.requires_grad)
    
    # Apply adaptation (freeze base)
    print("Applying personalization adaptation...")
    client.adapt_personalization(freeze_base=True)
    
    # Check requires_grad
    # In SimpleModel, params list is: fc1.weight, fc1.bias, fc2.weight, fc2.bias.
    # We freeze [:-2], so fc1 should be frozen, fc2 should be active.
    print(f"FC1 weight frozen: {not model.fc1.weight.requires_grad}")
    print(f"FC2 weight active: {model.fc2.weight.requires_grad}")
    
    assert not model.fc1.weight.requires_grad, "Base layer should be frozen"
    assert model.fc2.weight.requires_grad, "Head layer should be trainable"
    
    # Simulate training loop reset
    train_data = [{'inputs': torch.randn(5, 10), 'labels': torch.randint(0, 2, (5,))}]
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Simulating local training...")
    client.train_local(train_data, criterion, optimizer)
    
    # Check if reset happened
    print(f"FC1 weight active after training: {model.fc1.weight.requires_grad}")
    assert model.fc1.weight.requires_grad, "Model should be unfrozen after training"
    
    print("✓ Adaptive Personalization Test Passed")

if __name__ == "__main__":
    try:
        test_compression()
        test_continual_learning()
        test_adaptive_personalization()
        print("\n🎉 ALL TESTS PASSED")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
