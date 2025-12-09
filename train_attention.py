import os
import torch
torch.set_num_threads(8) 
import torch.nn as nn
import numpy as np
from data import create_dataloader
from validate import validate
from options.train_options import TrainOptions
from options.test_options import TestOptions


from networks.attention_npr import attention_npr_resnet50


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for img, label in loader:  
        img = img.to(device)
        label = label.to(device).float()

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output.squeeze(1), label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def test_model(model, test_opt):
    model.eval()
    with torch.no_grad():
        acc, ap, _, _, _, _ = validate(model, test_opt)
    return acc, ap


def train_attention_npr(num_epochs=30, batch_size=32, lr=0.0001, device='cuda'):
    opt = TrainOptions().parse()
    opt.batch_size = batch_size
    opt.lr = lr
    opt.niter = num_epochs

    train_dataroot = f'{opt.dataroot}/{opt.train_split}/'
    val_dataroot = f'{opt.dataroot}/{opt.val_split}/'

    opt.dataroot = train_dataroot
    train_loader = create_dataloader(opt)

    val_opt = TestOptions().parse(print_options=False)
    val_opt.dataroot = val_dataroot

    print("HIT: Attention model")
    model = attention_npr_resnet50(num_classes=1).to(device)  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"HIT: Training for {num_epochs} epochs")
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc, val_ap = test_model(model, val_opt)
        

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | AP: {val_ap:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_attention_model.pth')

    torch.save(model.state_dict(), 'final_attention_model.pth')
    print(f"Best accuracy: {best_acc:.4f}")
    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = train_attention_npr(
        num_epochs=30,   
        batch_size=16,  
        lr=0.0001,
        device=device
    )
