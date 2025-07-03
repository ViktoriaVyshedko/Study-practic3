import torch
import torch.nn as nn
import json


class FullyConnectedModel(nn.Module):
    def __init__(self, config_path=None, input_size=None, num_classes=None, **kwargs):
        super().__init__()
        
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = kwargs
        
        self.input_size = input_size or self.config.get('input_size', 784)
        self.num_classes = num_classes or self.config.get('num_classes', 10)
        
        self.layers = self._build_layers()
    
    def load_config(self, config_path):
        """Загружает конфигурацию из JSON файла"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _build_layers(self):
        layers = []
        prev_size = self.input_size
        
        layer_config = self.config.get('layers', [])
        
        for layer_spec in layer_config:
            layer_type = layer_spec['type']
            
            if layer_type == 'linear':
                out_size = layer_spec['size']
                layers.append(nn.Linear(prev_size, out_size))
                prev_size = out_size
                
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
                
            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())
                
            elif layer_type == 'tanh':
                layers.append(nn.Tanh())
                
            elif layer_type == 'dropout':
                rate = layer_spec.get('rate', 0.5)
                layers.append(nn.Dropout(rate))
                
            elif layer_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(prev_size))
                
            elif layer_type == 'layer_norm':
                layers.append(nn.LayerNorm(prev_size))
        
        layers.append(nn.Linear(prev_size, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


def create_model_from_config(config_path, input_size=None, num_classes=None):
    """Создает модель из JSON конфигурации"""
    return FullyConnectedModel(config_path, input_size, num_classes) 


import torch.optim as optim
from tqdm import tqdm


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        
        if not is_test and optimizer is not None:
            optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)


    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    } 


from homework.utils.utils import plot_training_history, count_parameters
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    class MNISTDataset(Dataset):
        def __init__(self, train=True, transform=None):
            super().__init__()
            self.dataset = torchvision.datasets.MNIST(
                root='./data', 
                train=train, 
                download=True, 
                transform=transform
            )

        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            return self.dataset[idx]

    class QMNISTDataset(Dataset):
        def __init__(self, train=True, transform=None):
            super().__init__()
            self.dataset = torchvision.datasets.QMNIST(
                root='./data', 
                train=train, 
                download=True, 
                transform=transform
            )
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            return self.dataset[idx]
    
    #train_dataset = MNISTDataset(train=True, transform=transform)
    #test_dataset = MNISTDataset(train=False, transform=transform)

    train_dataset = QMNISTDataset(train=True, transform=transform)
    test_dataset = QMNISTDataset(train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

train_loader, test_loader = get_mnist_loaders(batch_size=64)

model = FullyConnectedModel(
    input_size=784,
    num_classes=10,
    layers=[
        {"type": "linear", "size": 512},
        {"type": "batch_norm"},
        #{"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        #{"type": "sigmoid"},
        #{"type": "tanh"},
        #{"type": "layer_norm"},
        #{"type": "linear", "size": 256},
        #{"type": "relu"},
        #{"type": "dropout", "rate": 0.1},
        #{"type": "linear", "size": 128},
        #{"type": "relu"}
    ]
).to(device)

print(f"Model parameters: {count_parameters(model)}")

history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(1, 1)

epochs = range(1, len(history['train_losses']) + 1)
    
ax1.plot(epochs, history['train_accs'], label='Train Acc')
ax1.plot(epochs, history['test_accs'], label='Test Acc')
ax1.set_title('Accuracy')
ax1.legend()
    
plt.tight_layout()
plt.show()

plot_training_history(history) 