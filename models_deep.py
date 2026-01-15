import torch
import torch.nn as nn
import os

MODEL_PATH = 'models/deep_mlp.pth'

class SentimentMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(SentimentMLP, self).__init__()
        # Shared Layer
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Heads for each aspect
        # Each head takes the shared representation and predicts 3 classes (POS/NEU/NEG)
        self.head_food = nn.Linear(128, output_dim)
        self.head_service = nn.Linear(128, output_dim)
        self.head_price = nn.Linear(128, output_dim)
        self.head_global = nn.Linear(128, output_dim)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        shared = self.fc1(x)
        shared = self.relu(shared)
        shared = self.dropout(shared)
        
        out_food = self.head_food(shared)
        out_service = self.head_service(shared)
        out_price = self.head_price(shared)
        out_global = self.head_global(shared)
        
        return out_food, out_service, out_price, out_global

    def predict_proba(self, x):
        """Returns probabilities for all heads"""
        with torch.no_grad():
            f, s, p, g = self.forward(x)
            return {
                'food': self.softmax(f),
                'service': self.softmax(s),
                'price': self.softmax(p),
                'global': self.softmax(g)
            }

def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(input_dim, path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    model = SentimentMLP(input_dim)
    try:
        model.load_state_dict(torch.load(path))
    except:
        return None # Structure changed, need retraining
    model.eval()
    return model
