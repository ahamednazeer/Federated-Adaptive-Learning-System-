"""
Voice Encoder - CNN + Transformer for voice/audio encoding
Processes MFCC features to detect Parkinson's tremor patterns
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


class VoiceEncoder(nn.Module):
    """
    CNN + Transformer encoder for voice features
    Input: MFCC features (batch_size, num_mfcc, num_frames)
    Output: 128-dim embedding
    """
    
    def __init__(self, num_mfcc: int = 26, embedding_dim: int = 128):
        super(VoiceEncoder, self).__init__()
        
        self.num_mfcc = num_mfcc
        self.embedding_dim = embedding_dim
        
        # CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(num_mfcc, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Transformer for temporal modeling
        self.pos_encoder = PositionalEncoding(128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global pooling and projection
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
            x: (batch_size, num_mfcc, num_frames)
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, 128, num_frames/2)
        
        # Transpose for transformer (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, 128)
        
        # Projection to embedding
        embedding = self.fc(x)
        
        return embedding
    
    def encode_sample(self, mfcc_features: np.ndarray) -> np.ndarray:
        """
        Encode a single voice sample
        Args:
            mfcc_features: numpy array of shape (num_mfcc, num_frames)
        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(mfcc_features).unsqueeze(0)
            
            # Encode
            embedding = self.forward(x)
            
            return embedding.squeeze(0).numpy()


def create_voice_encoder(pretrained: bool = False) -> VoiceEncoder:
    """
    Factory function to create voice encoder
    Args:
        pretrained: Whether to load pretrained weights
    Returns:
        VoiceEncoder instance
    """
    model = VoiceEncoder()
    
    if pretrained:
        # TODO: Load pretrained weights from UCI training
        pass
    
    return model


if __name__ == '__main__':
    # Test the encoder
    print("Testing Voice Encoder...")
    
    # Create model
    encoder = create_voice_encoder()
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with random MFCC features
    mfcc = np.random.randn(26, 100)  # 26 MFCC coefficients, 100 frames
    embedding = encoder.encode_sample(mfcc)
    print(f"Input shape: {mfcc.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    
    # Test batch processing
    batch_size = 4
    batch = torch.randn(batch_size, 26, 100)
    output = encoder(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {output.shape}")
    
    print("✓ Voice Encoder test passed!")
