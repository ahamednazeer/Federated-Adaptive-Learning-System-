"""
Multi-Modal Fusion Layer
Combines embeddings from all modalities using attention mechanism
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for multi-modal embeddings
    Learns importance weights for each modality
    """
    
    def __init__(self, embedding_dim: int = 128, num_modalities: int = 5):
        super(AttentionFusion, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Fusion projection
        self.fusion_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
    
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        Args:
            embeddings: (batch_size, num_modalities, embedding_dim)
            mask: (batch_size, num_modalities) - 1 for available, 0 for missing
        Returns:
            fused: (batch_size, 256) - fused embedding
            attention_weights: (batch_size, num_modalities) - attention weights
        """
        # Compute attention scores
        attention_scores = self.attention(embeddings).squeeze(-1)  # (batch_size, num_modalities)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, num_modalities)
        
        # Weighted sum of embeddings
        fused = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, embedding_dim)
        
        # Project to fusion space
        fused = self.fusion_fc(fused)  # (batch_size, 256)
        
        return fused, attention_weights


class MultiModalFusion(nn.Module):
    """
    Complete multi-modal fusion system with disease-specific classifiers
    """
    
    def __init__(self, embedding_dim: int = 128):
        super(MultiModalFusion, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Attention-based fusion
        self.fusion = AttentionFusion(embedding_dim)
        
        # Disease-specific classifiers
        self.cvd_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.parkinsons_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.diabetes_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            embeddings: Dict with keys 'ecg', 'voice', 'handwriting', 'wearable', 'glucose'
                       Each value is (batch_size, embedding_dim)
        Returns:
            Dict with:
                - cvd_risk: (batch_size, 1)
                - parkinsons_risk: (batch_size, 1)
                - diabetes_risk: (batch_size, 1)
                - fusion_embedding: (batch_size, 256)
                - attention_weights: (batch_size, 5)
        """
        # Stack embeddings
        modality_order = ['ecg', 'voice', 'handwriting', 'wearable', 'glucose']
        embedding_list = []
        mask_list = []
        
        for modality in modality_order:
            if modality in embeddings and embeddings[modality] is not None:
                embedding_list.append(embeddings[modality])
                mask_list.append(torch.ones(embeddings[modality].size(0), 1))
            else:
                # Use zero embedding for missing modality
                batch_size = list(embeddings.values())[0].size(0)
                embedding_list.append(torch.zeros(batch_size, self.embedding_dim))
                mask_list.append(torch.zeros(batch_size, 1))
        
        # Stack
        stacked_embeddings = torch.stack(embedding_list, dim=1)  # (batch_size, 5, embedding_dim)
        mask = torch.cat(mask_list, dim=1)  # (batch_size, 5)
        
        # Fuse
        fused, attention_weights = self.fusion(stacked_embeddings, mask)
        
        # Classify
        cvd_risk = self.cvd_classifier(fused)
        parkinsons_risk = self.parkinsons_classifier(fused)
        diabetes_risk = self.diabetes_classifier(fused)
        
        return {
            'cvd_risk': cvd_risk,
            'parkinsons_risk': parkinsons_risk,
            'diabetes_risk': diabetes_risk,
            'fusion_embedding': fused,
            'attention_weights': attention_weights
        }
    
    def predict(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Predict disease risks from embeddings
        Args:
            embeddings: Dict with numpy arrays
        Returns:
            Dict with risk scores and attention weights
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            tensor_embeddings = {}
            for modality, emb in embeddings.items():
                if emb is not None:
                    tensor_embeddings[modality] = torch.FloatTensor(emb).unsqueeze(0)
            
            # Forward pass
            output = self.forward(tensor_embeddings)
            
            # Convert to numpy
            result = {
                'cvd_risk': float(output['cvd_risk'].squeeze().numpy()),
                'parkinsons_risk': float(output['parkinsons_risk'].squeeze().numpy()),
                'diabetes_risk': float(output['diabetes_risk'].squeeze().numpy()),
                'attention_weights': {
                    'ecg': float(output['attention_weights'][0, 0].numpy()),
                    'voice': float(output['attention_weights'][0, 1].numpy()),
                    'handwriting': float(output['attention_weights'][0, 2].numpy()),
                    'wearable': float(output['attention_weights'][0, 3].numpy()),
                    'glucose': float(output['attention_weights'][0, 4].numpy())
                }
            }
            
            return result


def create_multimodal_fusion(pretrained: bool = False) -> MultiModalFusion:
    """
    Factory function to create multi-modal fusion model
    Args:
        pretrained: Whether to load pretrained weights
    Returns:
        MultiModalFusion instance
    """
    model = MultiModalFusion()
    
    if pretrained:
        # TODO: Load pretrained weights
        pass
    
    return model


if __name__ == '__main__':
    # Test the fusion model
    print("Testing Multi-Modal Fusion...")
    
    # Create model
    fusion = create_multimodal_fusion()
    print(f"Model parameters: {sum(p.numel() for p in fusion.parameters()):,}")
    
    # Test with random embeddings
    batch_size = 4
    embeddings = {
        'ecg': torch.randn(batch_size, 128),
        'voice': torch.randn(batch_size, 128),
        'handwriting': torch.randn(batch_size, 128),
        'wearable': torch.randn(batch_size, 128),
        'glucose': torch.randn(batch_size, 128)
    }
    
    output = fusion(embeddings)
    
    print(f"CVD risk shape: {output['cvd_risk'].shape}")
    print(f"Parkinson's risk shape: {output['parkinsons_risk'].shape}")
    print(f"Diabetes risk shape: {output['diabetes_risk'].shape}")
    print(f"Fusion embedding shape: {output['fusion_embedding'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")
    
    # Test prediction with single sample
    single_embeddings = {
        'ecg': np.random.randn(128),
        'voice': np.random.randn(128),
        'handwriting': np.random.randn(128),
        'wearable': np.random.randn(128),
        'glucose': np.random.randn(128)
    }
    
    result = fusion.predict(single_embeddings)
    print(f"\nPrediction result:")
    print(f"  CVD risk: {result['cvd_risk']:.3f}")
    print(f"  Parkinson's risk: {result['parkinsons_risk']:.3f}")
    print(f"  Diabetes risk: {result['diabetes_risk']:.3f}")
    print(f"  Attention weights: {result['attention_weights']}")
    
    print("\n✓ Multi-Modal Fusion test passed!")
