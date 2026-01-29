"""
Handwriting Encoder - CNN + Bi-LSTM for handwriting encoding
Processes handwriting strokes to detect tremor and motor control issues
"""

import torch
import torch.nn as nn
import numpy as np


class HandwritingEncoder(nn.Module):
    """
    CNN + Bi-LSTM encoder for handwriting features
    Input: Handwriting features (batch_size, num_features, num_points)
    Output: 128-dim embedding
    """
    
    def __init__(self, num_features: int = 5, embedding_dim: int = 128):
        super(HandwritingEncoder, self).__init__()
        
        self.num_features = num_features  # x, y, velocity, acceleration, pressure
        self.embedding_dim = embedding_dim
        
        # CNN for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Bi-LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),  # 128 = 64*2 (bidirectional)
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Final projection
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
            x: (batch_size, num_features, num_points)
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, 64, num_points/2)
        
        # Transpose for LSTM (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, 128)
        
        # Attention pooling
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        x = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, 128)
        
        # Final projection
        embedding = self.fc(x)
        
        return embedding
    
    def encode_sample(self, handwriting_features: np.ndarray) -> np.ndarray:
        """
        Encode a single handwriting sample
        Args:
            handwriting_features: numpy array of shape (num_features, num_points)
        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(handwriting_features).unsqueeze(0)
            
            # Encode
            embedding = self.forward(x)
            
            return embedding.squeeze(0).numpy()


def create_handwriting_encoder(pretrained: bool = False) -> HandwritingEncoder:
    """
    Factory function to create handwriting encoder
    Args:
        pretrained: Whether to load pretrained weights
    Returns:
        HandwritingEncoder instance
    """
    model = HandwritingEncoder()
    
    if pretrained:
        # TODO: Load pretrained weights from HandPD training
        pass
    
    return model


if __name__ == '__main__':
    # Test the encoder
    print("Testing Handwriting Encoder...")
    
    # Create model
    encoder = create_handwriting_encoder()
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with random handwriting features
    features = np.random.randn(5, 500)  # 5 features, 500 points
    embedding = encoder.encode_sample(features)
    print(f"Input shape: {features.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    
    # Test batch processing
    batch_size = 4
    batch = torch.randn(batch_size, 5, 500)
    output = encoder(batch)
    print(f"Batch input shape: {batch.shape}")
    print(f"Batch output shape: {output.shape}")
    
    print("✓ Handwriting Encoder test passed!")
