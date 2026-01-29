"""
Federated Learning Server
Coordinates federated training rounds and aggregates model updates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json


class FederatedServer:
    """
    Federated learning server implementing FedAvg algorithm
    """
    
    def __init__(self, global_model: nn.Module, num_clients: int = 5):
        """
        Args:
            global_model: The global model to train
            num_clients: Number of federated clients (datasets)
        """
        self.global_model = global_model
        self.num_clients = num_clients
        self.current_round = 0
        self.client_updates = {}
        self.round_metrics = []
        
        # Client mapping to datasets
        self.client_datasets = {
            'client_0': 'PTB-XL',
            'client_1': 'UCI',
            'client_2': 'HandPD',
            'client_3': 'WESAD',
            'client_4': 'OhioT1DM'
        }
    
    def start_round(self) -> Dict:
        """
        Start a new federated learning round
        
        Returns:
            Dict with round info and global model weights
        """
        self.current_round += 1
        self.client_updates = {}
        
        # Get current global model weights
        global_weights = self.get_model_weights()
        
        round_info = {
            'round_number': self.current_round,
            'num_clients': self.num_clients,
            'global_weights': global_weights,
            'started_at': datetime.now().isoformat()
        }
        
        return round_info
    
    def receive_client_update(self, client_id: str, update: Dict):
        """
        Receive model update from a client
        
        Args:
            client_id: Client identifier
            update: Dict with 'gradients', 'loss', 'accuracy', 'num_samples'
        """
        self.client_updates[client_id] = {
            'gradients': update['gradients'],
            'loss': update.get('loss', 0.0),
            'accuracy': update.get('accuracy', 0.0),
            'num_samples': update.get('num_samples', 1),
            'dataset': self.client_datasets.get(client_id, 'unknown'),
            'received_at': datetime.now().isoformat()
        }
    
    def aggregate_updates(self) -> Dict:
        """
        Aggregate client updates using FedAvg
        
        Returns:
            Dict with aggregation results
        """
        if not self.client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in self.client_updates.values())
        
        # Weighted average of gradients
        aggregated_gradients = {}
        
        # Get first client's gradients as template
        first_client = list(self.client_updates.keys())[0]
        gradient_keys = self.client_updates[first_client]['gradients'].keys()
        
        for key in gradient_keys:
            weighted_sum = None
            
            for client_id, update in self.client_updates.items():
                weight = update['num_samples'] / total_samples
                client_grad = update['gradients'][key]
                
                if weighted_sum is None:
                    weighted_sum = client_grad * weight
                else:
                    weighted_sum += client_grad * weight
            
            aggregated_gradients[key] = weighted_sum
        
        # Calculate average metrics
        avg_loss = sum(u['loss'] * u['num_samples'] for u in self.client_updates.values()) / total_samples
        avg_accuracy = sum(u['accuracy'] * u['num_samples'] for u in self.client_updates.values()) / total_samples
        
        # Update global model
        self.apply_gradients(aggregated_gradients)
        
        # Record metrics
        round_result = {
            'round_number': self.current_round,
            'num_clients': len(self.client_updates),
            'total_samples': total_samples,
            'global_loss': float(avg_loss),
            'global_accuracy': float(avg_accuracy),
            'completed_at': datetime.now().isoformat()
        }
        
        self.round_metrics.append(round_result)
        
        return round_result
    
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current global model weights"""
        weights = {}
        for name, param in self.global_model.named_parameters():
            weights[name] = param.data.cpu().numpy()
        return weights
    
    def apply_gradients(self, gradients: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """
        Apply aggregated gradients to global model
        
        Args:
            gradients: Dict of aggregated gradients
            learning_rate: Learning rate for update
        """
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in gradients:
                    grad = gradients[name]
                    if isinstance(grad, np.ndarray):
                        grad = torch.from_numpy(grad)
                    param.data -= learning_rate * grad
    
    def get_round_metrics(self) -> List[Dict]:
        """Get metrics from all rounds"""
        return self.round_metrics
    
    def save_global_model(self, path: str):
        """Save global model to file"""
        torch.save(self.global_model.state_dict(), path)
    
    def load_global_model(self, path: str):
        """Load global model from file"""
        self.global_model.load_state_dict(torch.load(path))


class FederatedClient:
    """
    Federated learning client (edge device simulator)
    """
    
    def __init__(self, client_id: str, model: nn.Module, dataset_name: str):
        """
        Args:
            client_id: Client identifier
            model: Local model (copy of global model)
            dataset_name: Dataset assigned to this client
        """
        self.client_id = client_id
        self.model = model
        self.dataset_name = dataset_name
        self.local_epochs = 1
    
    def update_model(self, global_weights: Dict[str, np.ndarray]):
        """
        Update local model with global weights
        
        Args:
            global_weights: Global model weights from server
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_weights:
                    param.data = torch.from_numpy(global_weights[name])
    
    def train_local(self, train_data: List, criterion, optimizer) -> Dict:
        """
        Train local model on client data
        
        Args:
            train_data: List of training samples
            criterion: Loss function
            optimizer: Optimizer
        
        Returns:
            Dict with training results
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            for batch in train_data:
                # Forward pass
                outputs = self.model(batch['inputs'])
                loss = criterion(outputs, batch['labels'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                total += batch['labels'].size(0)
                
                # Accuracy (for classification)
                if hasattr(outputs, 'argmax'):
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == batch['labels']).sum().item()
        
        avg_loss = total_loss / len(train_data)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total
        }
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get current model gradients
        
        Returns:
            Dict of gradients
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients
    
    def create_update(self, train_results: Dict, gradients: Dict) -> Dict:
        """
        Create update package for server
        
        Args:
            train_results: Results from local training
            gradients: Model gradients
        
        Returns:
            Update dict
        """
        return {
            'client_id': self.client_id,
            'dataset': self.dataset_name,
            'gradients': gradients,
            'loss': train_results['loss'],
            'accuracy': train_results['accuracy'],
            'num_samples': train_results['num_samples']
        }


def create_federated_server(model: nn.Module, num_clients: int = 5) -> FederatedServer:
    """Factory function to create federated server"""
    return FederatedServer(model, num_clients)


def create_federated_client(client_id: str, model: nn.Module, 
                           dataset_name: str) -> FederatedClient:
    """Factory function to create federated client"""
    return FederatedClient(client_id, model, dataset_name)


if __name__ == '__main__':
    # Test federated learning
    print("Testing Federated Learning System...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    # Create server
    global_model = SimpleModel()
    server = create_federated_server(global_model, num_clients=5)
    print(f"Created server with {server.num_clients} clients")
    
    # Start round
    round_info = server.start_round()
    print(f"\nStarted round {round_info['round_number']}")
    
    # Simulate client updates
    for i in range(5):
        client_id = f"client_{i}"
        dataset_name = server.client_datasets[client_id]
        
        # Create client
        client_model = SimpleModel()
        client = create_federated_client(client_id, client_model, dataset_name)
        
        # Update with global weights
        client.update_model(round_info['global_weights'])
        
        # Simulate training (create fake gradients)
        fake_gradients = {
            'fc.weight': torch.randn(2, 10) * 0.01,
            'fc.bias': torch.randn(2) * 0.01
        }
        
        update = {
            'gradients': fake_gradients,
            'loss': np.random.uniform(0.5, 1.0),
            'accuracy': np.random.uniform(0.6, 0.9),
            'num_samples': np.random.randint(50, 200)
        }
        
        server.receive_client_update(client_id, update)
        print(f"  Received update from {client_id} ({dataset_name})")
    
    # Aggregate
    result = server.aggregate_updates()
    print(f"\nAggregation complete:")
    print(f"  Global loss: {result['global_loss']:.4f}")
    print(f"  Global accuracy: {result['global_accuracy']:.4f}")
    print(f"  Total samples: {result['total_samples']}")
    
    print("\n✓ Federated Learning test passed!")
