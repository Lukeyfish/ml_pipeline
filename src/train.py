import torch
import yaml

# Local Imports
from model.convnet import VGG16
from data import FashionDataset, FashionDataLoader
from trainer import Trainer

def main():

    # Assigns training to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('training on: ', device)

    # Opens the train config
    with open('./configs/main.yaml', 'r') as file:
        cfg = yaml.safe_load(file)


    # Instantiating the Train and Test sets
    train_set = FashionDataset(cfg['data']['train_ft'], cfg['data']['train_tg'])
    test_set = FashionDataset(cfg['data']['test_ft'], cfg['data']['test_tg'])

    in_channels, out_channels = train_set.get_in_out_size()

    # Instantiating the Model
    model = VGG16(in_channels, out_channels, cfg['model']['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['model']['lr'])
    
    dl = FashionDataLoader(train_set, cfg['trainer']['batch_size'], cfg['trainer']['shuffle'])
    
    # Instantiating the Trainer
    trainer = Trainer(train_set, dl.load(), model, optimizer, device)
    
    # Training the model
    trainer.train()
    

if __name__=="__main__":
    main()
