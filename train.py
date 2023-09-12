from typing import Any, List, Sequence, Union

import os
import glob
import wandb
import random
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, resnet50
import torchvision.transforms.functional as F

class Trainer:
    def __init__(self, 
                 model_name: str,
                 lr: float,
                 epochs: int,
                 batch_size: int,
                 num_classes: int,
                 num_workers: int,
                 device: torch.device,
                 device_ids: Sequence[int],
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 log_dir: str,
                 project_name: str,
                 wandb_name: str,
                 model_path: str = None,
                 use_wandb: bool = True,
                 use_amp: bool = True,
                ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = os.path.join(log_dir, project_name)
        self.project_name = project_name
        self.model_path = model_path
        self.use_amp = use_amp
        self.use_wandb = use_wandb
        self.wandb_id = None
        self.loss = 0
        self.best_acc = 0
        self.device = torch.device(device)
        
        if model_name == "resnet34":
            self.model = resnet34()
        elif model_name == "resnet50":
            self.model = resnet50()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(self.device)
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=epochs // 20)
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
        
        if device_ids:
            self.model = DataParallel(self.model, device_ids)
            
        if use_wandb:
            if model_path is None:
                if wandb_name is None: wandb_name = log_dir
                wandb.init(project=project_name, 
                           name=f"{wandb_name}",
                           config=self.__dict__)
            else:
                assert self.wandb_id != 0
                wandb.init(project=project_name, 
                           id=self.wandb_id, 
                           resume=True)
    
    def train(self):
        for epoch in range(self.epochs):
            count = 0
            running_loss = 0.0
            self.model.train()
            
            with torch.cuda.amp.autocast(self.use_amp):
                with tqdm(self.train_loader, desc=f"train {epoch+1}/{self.epochs}") as t:
                    
                    for inputs, labels in self.train_loader:
                        count += inputs.size(0)
                        self.optimizer.zero_grad()
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                        
                        running_loss += loss.item()
                        
                        t.set_postfix(loss=running_loss / count, lr=self.optimizer.param_groups[0]['lr'])
                        t.update(1)

                self.scheduler.step()
                self.loss = running_loss / len(self.train_loader)
                if self.use_wandb: wandb.log({"loss": self.loss}, epoch)
                self.save_model(self.model, 
                                self.optimizer, 
                                self.scheduler,
                                epoch,
                                os.path.join(self.log_dir, "last.pt"))
                self.validate(epoch)

    def validate(self, epoch):
        total = 0
        correct = 0
        self.model.eval()
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"val {epoch+1}/{self.epochs}") as t:
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == torch.argmax(labels, 1)).sum().item()

                    t.update(1)
                        
        accuracy = correct / total
        
        if self.use_wandb: wandb.log({"accuracy": accuracy}, epoch)
        
        if self.best_acc < accuracy:
            self.best_acc = accuracy
            self.save_model(self.model, 
                            self.optimizer, 
                            self.scheduler,
                            epoch,
                            os.path.join(self.log_dir, "best.pt"))
        
        print(f"Validation Accuracy : {accuracy*100:.2f}%")
    
    def save_model(
        self,
        model, 
        optimizer, 
        scheduler, 
        epoch, 
        save_path,
    ):
        save_dir = os.path.dirname(save_path)

        os.makedirs(save_dir, exist_ok=True)
        
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch+1,
            'loss': self.loss,
            'best_acc': self.best_acc,
            'project_name': self.project_name,
            'id': wandb.run.id if self.use_wandb else 0,
        }
        
        torch.save(state, save_path)

        print(f"model is saved in {save_path}")
        
    
    def load_checkpoint(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.start_epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.best_mean_dice = state_dict['best_mean_dice']
        self.wandb_id = state_dict['id']
        
        print(f"Checkpoint loaded from {model_path}")
        
        
class LocationDataset(Dataset):
    def __init__(self,
                 image_path: str,
                 imgsz: int,
                 num_classes: int,
                 transform: transforms = None
                 ) -> None:
        super().__init__()
        self.image_path = image_path
        self.imgsz = imgsz 
        self.num_classes = num_classes
        self.transform = transform
        
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((imgsz, imgsz))
        
        self.images = []
        
        for i in range(num_classes):
            self.images += glob.glob(os.path.join(self.image_path, str(i), "*.jpg"))
            
        self.cache = {}
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, i: int) -> Any:
        if self.images[i] not in self.cache.keys():
            image = Image.open(self.images[i])
            image = self.to_tensor(image).type(torch.uint8)
            # image = self.crop(image).type(torch.uint8)
            self.cache[self.images[i]] = image
        else:
            image = self.cache[self.images[i]]
        
        label = int(os.path.basename(os.path.dirname(self.images[i])))
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.resize(image)
        
        return image.float(), torch.eye(self.num_classes)[label] # .float()
    
    def crop(self, image: torch.Tensor, r: float = 0.1):
        _, h, w = image.shape
        # return image[:, :, :int(h*0.5), int(w*r):int(w*(1-r))]
        return image[:, :int(h*0.5), :]
    

if __name__ == "__main__":
    model_name = "resnet34"
    lr = 0.0005
    epochs = 200
    batch_size = 400
    num_classes = 8
    num_workers = 2
    device = "cuda:0"
    device_ids = [i for i in range(torch.cuda.device_count())]
    log_dir = "logs"
    project_name = "place_recognition"
    wandb_name = "1"
    
    train_path = "datasets/pre/train"
    val_path = "datasets/pre/val"
    imgsz = 512
    
    transform = transforms.Compose([
        transforms.RandomAffine((-10, 10), (0.05, 0.1)),
        # transforms.RandomResizedCrop((imgsz, imgsz), (0.9, 1)),
        transforms.RandomErasing(0.4, (0.02, 0.15)),
        transforms.RandomEqualize(),
        transforms.ColorJitter(0.2, 0.3),
    ])
    
    train_dataset = LocationDataset(train_path, imgsz, num_classes, transform)
    val_dataset = LocationDataset(val_path, imgsz, num_classes)
    
    trainer = Trainer(
        model_name=model_name,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        num_classes=num_classes,
        num_workers=num_workers,
        device=device,
        device_ids=device_ids,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        log_dir=log_dir,
        project_name=project_name,
        wandb_name=wandb_name,
    )
    
    trainer.train()