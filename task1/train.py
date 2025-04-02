import torch
from torch.utils.data import DataLoader
from utils.cnn_rnn import CNN_RNN_Model
from utils.Prepare_data import WikiArtDataset
from utils.trainer import Trainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = WikiArtDataset(csv_file="/atharvamalode/dca5b7c5-7f3f-4cbd-b82d-03706cafddf6/Atharva/HumanAI/task1/dataset/csv/genre.csv")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    input_size = 8 + 1024  
    
    model = CNN_RNN_Model(input_size=input_size, hidden_size=256, num_classes=23)
    
    trainer = Trainer(model, train_loader, device, lr=1e-4, epochs=20, save_dir="weights")
    
    trainer.train()

if __name__ == "__main__":
    main()