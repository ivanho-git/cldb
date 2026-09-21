import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from cldb.cl_engine import EMBEDDING_DIM, CONTEXT_DIM


class BaselinePeriodic:
    def __init__(self, feature_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.embedding_dim = EMBEDDING_DIM
        self.context_dim = CONTEXT_DIM
        self.total_dim = feature_dim + EMBEDDING_DIM + CONTEXT_DIM

        self.model = nn.Sequential(
            nn.Linear(self.total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train_on_batch(self, x_features, x_embeddings, x_context, y_tensor):
        self.model.train()
        x_combined = torch.cat([x_features, x_embeddings, x_context], dim=1)
        dataset = TensorDataset(x_combined, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(5):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, x_features, x_embeddings, x_context):
        self.model.eval()
        with torch.no_grad():
            x_combined = torch.cat([x_features, x_embeddings, x_context], dim=1)
            logits = self.model(x_combined.to(self.device))
            probs = torch.softmax(logits, dim=1)
            confidence, action_idx = torch.max(probs, dim=1)
        return action_idx.cpu().numpy(), confidence.cpu().numpy()