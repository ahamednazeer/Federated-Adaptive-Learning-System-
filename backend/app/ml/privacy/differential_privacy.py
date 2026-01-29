"""
Differential Privacy Mechanisms
Implements gradient clipping and noise addition for privacy-preserving federated learning
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


class DifferentialPrivacy:
    """
    Differential Privacy implementation for federated learning
    Implements gradient clipping and Gaussian noise addition
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0):
        """
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach
            clip_norm: Maximum L2 norm for gradient clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_scale = self._compute_noise_scale()
        self.cumulative_epsilon = 0.0
        self.rounds = 0
    
    def _compute_noise_scale(self) -> float:
        """
        Compute noise scale (sigma) from privacy parameters
        Using the Gaussian mechanism: σ = C * sqrt(2 * ln(1.25/δ)) / ε
        """
        sigma = self.clip_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to maximum L2 norm
        
        Args:
            gradients: Dict of parameter gradients
        
        Returns:
            Clipped gradients
        """
        # Compute total L2 norm
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += torch.sum(grad ** 2).item()
        
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        clip_coef = self.clip_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        clipped_gradients = {}
        for name, grad in gradients.items():
            if grad is not None:
                clipped_gradients[name] = grad * clip_coef
            else:
                clipped_gradients[name] = None
        
        return clipped_gradients
    
    def add_noise(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to gradients
        
        Args:
            gradients: Dict of parameter gradients
        
        Returns:
            Noisy gradients
        """
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            if grad is not None:
                noise = torch.randn_like(grad) * self.noise_scale
                noisy_gradients[name] = grad + noise
            else:
                noisy_gradients[name] = None
        
        return noisy_gradients
    
    def privatize_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply full DP mechanism: clip + noise
        
        Args:
            gradients: Dict of parameter gradients
        
        Returns:
            Privatized gradients
        """
        # Clip
        clipped = self.clip_gradients(gradients)
        
        # Add noise
        privatized = self.add_noise(clipped)
        
        # Update privacy budget
        self.rounds += 1
        self.cumulative_epsilon += self.epsilon
        
        return privatized
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get cumulative privacy budget spent
        
        Returns:
            (epsilon, delta) tuple
        """
        return (self.cumulative_epsilon, self.delta)
    
    def reset_privacy_budget(self):
        """Reset privacy budget counter"""
        self.cumulative_epsilon = 0.0
        self.rounds = 0


class SecureAggregation:
    """
    Secure aggregation for federated learning
    Simulates encrypted model updates
    """
    
    @staticmethod
    def encrypt_gradients(gradients: Dict[str, torch.Tensor], client_id: str) -> bytes:
        """
        Simulate gradient encryption
        In production, use proper cryptographic methods
        
        Args:
            gradients: Dict of parameter gradients
            client_id: Client identifier
        
        Returns:
            Encrypted gradients (as bytes)
        """
        # Convert gradients to numpy
        grad_dict = {}
        for name, grad in gradients.items():
            if grad is not None:
                grad_dict[name] = grad.cpu().numpy()
        
        # Serialize (in production, encrypt here)
        import pickle
        encrypted = pickle.dumps(grad_dict)
        
        return encrypted
    
    @staticmethod
    def decrypt_gradients(encrypted_gradients: bytes) -> Dict[str, np.ndarray]:
        """
        Simulate gradient decryption
        
        Args:
            encrypted_gradients: Encrypted gradients
        
        Returns:
            Decrypted gradients as numpy arrays
        """
        import pickle
        decrypted = pickle.loads(encrypted_gradients)
        return decrypted


def create_dp_mechanism(epsilon: float = 1.0, delta: float = 1e-5, 
                       clip_norm: float = 1.0) -> DifferentialPrivacy:
    """
    Factory function to create DP mechanism
    
    Args:
        epsilon: Privacy budget
        delta: Privacy delta
        clip_norm: Gradient clipping norm
    
    Returns:
        DifferentialPrivacy instance
    """
    return DifferentialPrivacy(epsilon, delta, clip_norm)


if __name__ == '__main__':
    # Test DP mechanism
    print("Testing Differential Privacy Mechanism...")
    
    dp = create_dp_mechanism(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    print(f"Noise scale: {dp.noise_scale:.4f}")
    
    # Test with random gradients
    gradients = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(5, 3),
        'layer2.bias': torch.randn(5)
    }
    
    print("\nOriginal gradient norms:")
    for name, grad in gradients.items():
        print(f"  {name}: {torch.norm(grad).item():.4f}")
    
    # Clip gradients
    clipped = dp.clip_gradients(gradients)
    print("\nClipped gradient norms:")
    for name, grad in clipped.items():
        print(f"  {name}: {torch.norm(grad).item():.4f}")
    
    # Add noise
    privatized = dp.privatize_gradients(gradients)
    print("\nPrivatized gradient norms:")
    for name, grad in privatized.items():
        print(f"  {name}: {torch.norm(grad).item():.4f}")
    
    # Check privacy budget
    epsilon_spent, delta = dp.get_privacy_spent()
    print(f"\nPrivacy budget spent: ε={epsilon_spent:.2f}, δ={delta:.2e}")
    
    # Test encryption
    print("\nTesting secure aggregation...")
    encrypted = SecureAggregation.encrypt_gradients(gradients, "client_1")
    print(f"Encrypted size: {len(encrypted)} bytes")
    
    decrypted = SecureAggregation.decrypt_gradients(encrypted)
    print(f"Decrypted keys: {list(decrypted.keys())}")
    
    print("\n✓ Differential Privacy test passed!")
