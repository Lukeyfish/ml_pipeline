import torch
import yaml
from torch.utils.data import Subset

# Local Imports
from model.convnet import VGG11, VGG16, BasicCNN
from data import FashionDataset, FashionDataLoader
from trainer import Trainer, Tester

def main():

    # Assigns training to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('training on: ', device)

    # Opens the train config
    with open('./configs/main.yaml', 'r') as file:
        cfg = yaml.safe_load(file)


    # Instantiating the Train and Test sets
    train_set = FashionDataset(cfg['data']['train_ft'], cfg['data']['train_tg'])

    # Splitting the training set into training and validation sets
    train_indices = list(range(0, 50000))
    val_indices = list(range(50000, len(train_set)))

    val_set = Subset(train_set, val_indices)
    train_set = Subset(train_set, train_indices)

    test_set = FashionDataset(cfg['data']['test_ft'], cfg['data']['test_tg'])
    

    # Instantiating the Model
    model = BasicCNN(cfg['model']['dropout_rate']).to(device)
    
    # Instantiating the Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['trainer']['optim']['lr'])
    
    train_loader = FashionDataLoader(train_set, cfg['trainer']['batch_size'], cfg['trainer']['shuffle'])
    val_loader = FashionDataLoader(val_set, cfg['trainer']['batch_size'], cfg['trainer']['shuffle'])
    test_loader = FashionDataLoader(test_set, cfg['trainer']['batch_size'], cfg['trainer']['shuffle'])
    
    # Instantiating the Trainer and Tester
    trainer = Trainer(
            train_loader.load(), 
            val_loader.load(), 
            model, 
            optimizer, 
            device, 
            cfg['model']['save_dir'],
            cfg['model']['save_name'],
            cfg['trainer']['num_epochs']
        )
    tester = Tester(test_loader.load(), model, device)
    
    # Training the model
    trainer.train()
    
    # Evaluating the models performance
    print(f"Test Accuracy: {tester.test():.2f}%")

    

if __name__=="__main__":
    main()
