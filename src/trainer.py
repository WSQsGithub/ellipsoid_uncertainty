import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, points = data.to(self.device), target.to(self.device) # data: (batch_size, 19), points: (batch_size, sample_size, 3)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, points)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

        return {"train_loss": total_loss / num_batches}

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return {"val_loss": total_loss / num_batches}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["train_loss"])

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_metrics['train_loss']:.4f}")

            if save_path is not None:
                torch.save(self.model.state_dict(), save_path)

        return history