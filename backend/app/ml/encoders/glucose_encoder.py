"""
Glucose Encoder - Temporal Transformer for CGM encoding
Processes continuous glucose monitoring time series
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class GlucoseEncoder(nn.Module):
    """
    Temporal Transformer encoder for glucose time series
    Input: Glucose readings (batch_size, seq_len)
    Output: 128-dim embedding
    """
    
    def __init__(self, embedding_dim: int = 128):
        super(GlucoseEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.input_projection = nn.Linear(1, 128)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(128, max_len=500)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Output projection
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
            x: (batch_size, seq_len) or (batch_size, seq_len, 1)
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, 128)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, 128)
        
        # Output projection
        embedding = self.fc(x)
        
        return embedding
    
    def encode_sample(self, glucose_readings: np.ndarray) -> np.ndarray:
        """
        Encode a single glucose sample
        Args:
            glucose_readings: numpy array of shape (seq_len,)
        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(glucose_readings).unsqueeze(0)
            
            # Encode
            embedding = self.forward(x)
            
            return embedding.squeeze(0).numpy()


def create_glucose_encoder(pretrained: bool = False) -> GlucoseEncoder:
    """
    Factory function to create glucose encoder
    Args:
        pretrained: Whether to load pretrained weights
    Returns:
        GlucoseEncoder instance
    """
    model = GlucoseEncoder()
    
    if pretrained:
        # TODO: Load pretrained weights from OhioT1DM training
        pass
    
    return model


if __name__ == '__main__':
    # Test the encoder
    print("Testing Glucose Encoder...")
    
    # Create model
    encoder = create_glucose_encoder()
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with random glucose readings
    glucose = np.random.randn(288)  # 288 readings (24 hours @ 5-min intervals)
    embedding = encoder.encode_sample(glucose)
    print(f"Input shape: {glucose.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    
    # Test batch processing
    batch_size = 4
    batch = torch.randn(batch_size, 288)
    output = encoder(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {output.shape}")
    
    print("✓ Glucose Encoder test passed!")
