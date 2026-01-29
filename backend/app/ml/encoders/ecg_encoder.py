"""
ECG Encoder - 1D CNN for ECG waveform encoding
Processes ECG signals to extract cardiac health features
"""

import torch
import torch.nn as nn
import numpy as np


class ECGEncoder(nn.Module):
    """
    1D CNN encoder for ECG signals
    Input: ECG waveform (batch_size, 1, 5000)
    Output: 128-dim embedding
    """
    
    def __init__(self, input_length: int = 5000, embedding_dim: int = 128):
        super(ECGEncoder, self).__init__()
        
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=50, stride=2, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=10, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened size
        self._calculate_flatten_size()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
    
    def _calculate_flatten_size(self):
        """Calculate the size after convolutions"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_length)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flatten_size = x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, 1, input_length) or (batch_size, input_length)
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        embedding = self.fc(x)
        
        return embedding
    
    def encode_sample(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Encode a single ECG sample
        Args:
            ecg_signal: numpy array of shape (input_length,)
        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(ecg_signal).unsqueeze(0).unsqueeze(0)
            
            # Pad or truncate to input_length
            if x.size(2) < self.input_length:
                padding = self.input_length - x.size(2)
                x = torch.nn.functional.pad(x, (0, padding))
            elif x.size(2) > self.input_length:
                x = x[:, :, :self.input_length]
            
            # Encode
            embedding = self.forward(x)
            
            return embedding.squeeze(0).numpy()


def create_ecg_encoder(pretrained: bool = False) -> ECGEncoder:
    """
    Factory function to create ECG encoder
    Args:
        pretrained: Whether to load pretrained weights
    Returns:
        ECGEncoder instance
    """
    model = ECGEncoder()
    
    if pretrained:
        # TODO: Load pretrained weights from PTB-XL training
        pass
    
    return model


if __name__ == '__main__':
    # Test the encoder
    print("Testing ECG Encoder...")
    
    # Create model
    encoder = create_ecg_encoder()
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with random ECG signal
    ecg_signal = np.random.randn(5000)
    embedding = encoder.encode_sample(ecg_signal)
    print(f"Input shape: {ecg_signal.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding dim: {embedding.shape[0]}")
    
    # Test batch processing
    batch_size = 4
    batch = torch.randn(batch_size, 1, 5000)
    output = encoder(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {output.shape}")
    
    print("✓ ECG Encoder test passed!")
