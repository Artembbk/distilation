from tqdm import tqdm
from transformers import AutoModelForTokenClassification
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
import wandb
from metric import calc_f1

class Trainer():
    def __init__(self, model: AutoModelForTokenClassification, 
                 opt: Optimizer, 
                 train_loader: DataLoader, 
                 valid_loader: DataLoader, 
                 criterion: CrossEntropyLoss, 
                 num_epochs, 
                 device):
        self.model = model
        self.opt = opt
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = device

    def train(self):
        wandb.init("distilation")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            for step, batch in tqdm(enumerate(self.train_loader), desc="Train", total=len(self.train_loader)):
                labels = batch['labels'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                self.opt.zero_grad()    
                out = self.model(input_ids)
                loss = self.criterion(out.logits.view(-1, 9), labels.view(-1))
                loss.backward()
                self.opt.step()

                predictions = out.logits.argmax(-1).tolist()
                score = calc_f1(predictions, labels.tolist())
                
                if step % 50:
                    wandb.log({"train loss": loss})
                    wandb.log({"train score": score})

            self.model.eval()
            total_loss = 0
            total_score = 0
            for step, batch in tqdm(enumerate(self.valid_loader), desc="Val", total=len(self.train_loader)):
                labels = batch['labels'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)   
                out = self.model(input_ids)
                loss = self.criterion(out.logits, labels)

                predictions = out.logits.argmax(-1)
                score = calc_f1(predictions, labels)

                total_loss += loss.item() / len(self.valid_loader)
                total_score += score / len(self.valid_loader)
                
            wandb.log({"val loss": total_loss})
            wandb.log({"val score": total_score})

        wandb.finish()






