import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import clean_text
from features import fit_transform_text, save_vectorizer
from models_deep import SentimentMLP, save_model
import pickle
import os

# Hardcoded Label Mapping to prevent flips
LABEL_MAP = {'NEG': 0, 'NEU': 1, 'POS': 2}
class ExplicitLabelEncoder:
    def transform(self, series):
        return series.map(LABEL_MAP).values
    def inverse_transform(self, arr):
        inv_map = {v: k for k, v in LABEL_MAP.items()}
        return [inv_map.get(x, 'NEU') for x in arr]

def main():
    print("Loading data...")
    if not os.path.exists('data/train_labeled.csv'):
        print("Data not found.")
        return
    df = pd.read_csv('data/train_labeled.csv')

    print("Preprocessing...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    df = df[df['cleaned_text'] != ""]

    print("Vectorizing...")
    # Use bigrams to capture "no rico", "muy bueno"
    X, vectorizer = fit_transform_text(df['cleaned_text'], vectorizer_type='tfidf') 
    save_vectorizer(vectorizer, 'models/vectorizer_tfidf.pkl')

    # Encode Labels Explicitly
    le = ExplicitLabelEncoder()
    
    y_food = le.transform(df['sentiment_food'])
    y_service = le.transform(df['sentiment_service'])
    y_price = le.transform(df['sentiment_price'])
    y_global = le.transform(df['sentiment_global'])
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # Convert to Tensors
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
    y_f_t = torch.tensor(y_food, dtype=torch.long)
    y_s_t = torch.tensor(y_service, dtype=torch.long)
    y_p_t = torch.tensor(y_price, dtype=torch.long)
    y_g_t = torch.tensor(y_global, dtype=torch.long)

    # Dataset & Loader
    dataset = TensorDataset(X_tensor, y_f_t, y_s_t, y_p_t, y_g_t)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X.shape[1]
    # Output dim is 3 (NEG, NEU, POS)
    model = SentimentMLP(input_dim, output_dim=3)
    
    criterion = nn.CrossEntropyLoss()
    # Lower Learning Rate slightly for stability
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 40 # Sufficient for synthetic data
    print(f"Training Multi-Head MLP for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_f, batch_s, batch_p, batch_g in dataloader:
            optimizer.zero_grad()
            
            out_f, out_s, out_p, out_g = model(batch_X)
            
            loss_f = criterion(out_f, batch_f)
            loss_s = criterion(out_s, batch_s)
            loss_p = criterion(out_p, batch_p)
            loss_g = criterion(out_g, batch_g)
            
            total_loss = loss_f + loss_s + loss_p + loss_g
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {epoch_loss:.4f}")

    print("Saving model...")
    save_model(model, 'models/deep_mlp.pth')
    print("Deep Training complete.")

if __name__ == "__main__":
    main()
