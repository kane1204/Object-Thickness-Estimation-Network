import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import time
import torch.optim as optim

from src.evaluation import accuracy_fast, mse_loss_with_nans

class Trainer():
    def __init__(self, model, optimiser, loss_fn, training_data, validation_data, acc_thresh=0.1, scheduler=None, view=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))   
        self.model = model.to(self.device)
        self.view = not view
        self.optimiser = optimiser
        self.scheduler = scheduler
        
        self.training_data = training_data
        self.validation_data = validation_data
        self.loss_fn = loss_fn
        self.loss_fn = self.loss_fn.to(self.device)
        self.acc_thresh = acc_thresh
    def train(self):#
        epoch_acc, epoch_loss = 0, 0
        batches = 0 
        self.model.train()

        for batch in tqdm(self.training_data, desc="Training Step", disable=self.view):
            x = batch['img'].to(self.device, dtype=torch.float)
            y = batch['thick_map'].to(self.device, dtype=torch.float)
            pred = self.model(x)

            loss = self.loss_fn(pred, y)
            train_acc = mse_loss_with_nans(pred, y)
            
            self.optimiser.zero_grad()
            loss.backward()
            
            self.optimiser.step()
            

            epoch_loss += loss.item()
            epoch_acc += train_acc
            batches += 1
            # Un indent after done for overfitting test
        train_acc = epoch_acc/batches
        train_loss = epoch_loss/batches
        return train_acc, train_loss
    
    def validate(self):
        epoch_acc, epoch_loss = 0, 0
        batches = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.validation_data, desc="Validation Step", disable=self.view):
                x = batch['img'].to(self.device, dtype=torch.float)
                y = batch['thick_map'].to(self.device, dtype=torch.float)

                pred = self.model(x)
                
                loss = self.loss_fn(pred, y)
                val_acc =  mse_loss_with_nans(pred, y)

                epoch_loss += loss.item()
                epoch_acc += val_acc
                batches += 1

            val_acc = epoch_acc/batches
            val_loss = epoch_loss/batches
            return val_acc, val_loss
        

    def run(self, epochs):
        save = True
        torch.cuda.empty_cache()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_desc = f"model_{timestr}"
        train_loss, train_acc, val_loss, val_acc = 0,0,0,0
        for epoch in tqdm(range(epochs), desc="Epochs"):
            train_acc, train_loss, val_acc, val_loss  = 0,0,0,0

            train_acc, train_loss = self.train()
            val_acc, val_loss = self.validate()
            if self.scheduler is not None:
                self.scheduler.step(train_loss)

            if save:
                # Append Training and validation stats to file
                self.append_file(f"{file_desc}_train_loss", train_loss)
                self.append_file(f"{file_desc}_train_masked_loss", train_acc)
                self.append_file(f"{file_desc}_val_loss", val_loss)
                self.append_file(f"{file_desc}_val_masked_loss", val_acc)
                path = f"models/{file_desc}_{epoch}"
                torch.save(self.model.state_dict(), path)
            print(f'Finished Epoch: {epoch} | Train Masked Loss: {train_acc:.5f} | Train Loss: {train_loss:.5f} | Val  Masked Loss: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
            # print(f'Finished Epoch: {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')

        print("Done!")
        return self.model
    
    def append_file(self, filename, data):
        file1 = open(f"results/{filename}.txt", "a")  # append mode
        file1.write(f"{data}\n")
        file1.close()