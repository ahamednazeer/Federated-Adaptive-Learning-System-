"""
Wearable Encoder - LSTM for wearable sensor encoding
Processes multi-sensor physiological signals
"""

import torch
import torch.nn as nn
import numpy as np


class WearableEncoder(nn.Module):
    """
    LSTM encoder for wearable sensor data
    Input: Multi-sensor features (batch_size, num_features, seq_len)
    Output: 128-dim embedding
    """
    
    def __init__(self, num_features: int = 5, embedding_dim: int = 128):
        super(WearableEncoder, self).__init__()
        
        self.num_features = num_features  # HR, SC, acc_x, acc_y, acc_z
        self.embedding_dim = embedding_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, num_features, seq_len)
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # Transpose for LSTM (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        x = h_n[-1]  # (batch_size, 128)
        
        # Fully connected
        embedding = self.fc(x)
        
        return embedding
    
    def encode_sample(self, wearable_features: np.ndarray) -> np.ndarray:
        """
        Encode a single wearable sample
        Args:
            wearable_features: numpy array of shape (num_features, seq_len)
        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(wearable_features).unsqueeze(0)
            
            # Encode
            embedding = self.forward(x)
            
            return embedding.squeeze(0).numpy()


def create_wearable_encoder(pretrained: bool = False) -> WearableEncoder:
    """
    Factory function to create wearable encoder
    Args:
        pretrained: Whether to load pretrained weights
    Returns:
        WearableEncoder instance
    """
    model = WearableEncoder()
    
    if pretrained:
        # TODO: Load pretrained weights from WESAD training
        pass
    
    return model


if __name__ == '__main__':
    # Test the encoder
    print("Testing Wearable Encoder...")
    
    # Create model
    encoder = create_wearable_encoder()
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with random wearable features
    features = np.random.randn(5, 960)  # 5 features, 960 samples (30s @ 32Hz)
    embedding = encoder.encode_sample(features)
    print(f"Input shape: {features.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    
    # Test batch processing
    batch_size = 4
    batch = torch.randn(batch_size, 5, 960)
    output = encoder(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {output.shape}")
    
    print("✓ Wearable Encoder test passed!")
