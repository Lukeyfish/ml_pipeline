from torch import nn
from tqdm import tqdm

# Trainer class to train the model and evaluate it
class Trainer:
    def __init__(
        self, 
        train_set, 
        train_loader, 
        model, 
        optimizer,
        device
    ):
        self.train_set = train_set
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train(self):
                
        # Set the model to training mode
        self.model.train()
        
        running_loss = 0.0

        for x, y in tqdm(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        tqdm.write(f"Training Loss: {running_loss / len(self.train_loader)}")
            