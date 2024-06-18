import os
import torch
from torch import nn
from tqdm import tqdm

# Trainer class to train the model and evaluate it
class Trainer:
    def __init__(
        self, 
        train_loader,
        val_loader, 
        model, 
        optimizer,
        device,
        save_path,
        save_name,
        num_epochs
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.save_name = save_name
        self.num_epochs = num_epochs
    
    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            
            # Set the model to training mode
            self.model.train()
        
            running_loss = 0.0
            train_loader_tqdm = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            for x, y in train_loader_tqdm:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = nn.CrossEntropyLoss()(y_hat, y)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                # Update tqdm with the running loss
                train_loader_tqdm.set_postfix(loss=loss.item())

            val_loss, val_accuracy = self.validate()
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # Save the model if the validation loss is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # If the save path does not exist, create it
                os.makedirs(self.save_path, exist_ok=True)
                
                torch.save(self.model.state_dict(), f"{self.save_path}{self.save_name}.pt")
                print(f"Saved model with validation loss {best_val_loss:.4f}")
    
                
    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = nn.CrossEntropyLoss()(y_hat, y)
                val_loss += loss.item()
                
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total
        
        return avg_val_loss, val_accuracy
    
    
class Tester:
    def __init__(
        self, 
        test_loader, 
        model, 
        device
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device
    
    def test(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        test_accuracy = 100 * correct / total
        
        return test_accuracy