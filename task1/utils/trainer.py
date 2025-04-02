import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  

class Trainer:
    """
    Trainer class to handle the training process of the CNN-RNN model.
    Handles loss computation, backpropagation, weight updates, and model saving.
    """
    def __init__(self, model, train_loader, device, lr=1e-4, epochs=10, save_dir="weights"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for sequences, labels in pbar:
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            avg_loss = total_loss / (pbar.n + 1)
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.4f}'})
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy
        
    def save_model(self, epoch):
        """Saves the model weights after each epoch."""
        save_path = os.path.join(self.save_dir, f"cnn_rnn_epoch_{epoch + 1}.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"[INFO] Model saved at {save_path}")
        
    def train(self):
        """Trains the model over multiple epochs."""
        print(f"[INFO] Starting training for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            self.save_model(epoch)
        
        print("\nTraining Completed! Model weights saved in 'weights/' directory.")
