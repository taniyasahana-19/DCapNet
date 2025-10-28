import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_unet(unet, train_loader, val_loader, device, epochs=50, lr=1e-3, patience=10):
    unet.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    best_val = float('inf')
    trigger=0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    for epoch in range(epochs):
        unet.train()
        running=0.0
        for batch in tqdm(train_loader, desc=f'UNet Train Epoch {epoch+1}/{epochs}'):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            preds = unet(imgs)
            loss = criterion(preds, masks)
            loss.backward(); optimizer.step()
            running += loss.item()
        train_loss = running/len(train_loader)
        # validation
        unet.eval()
        val_running=0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                preds = unet(imgs)
                loss = criterion(preds, masks)
                val_running += loss.item()
        val_loss = val_running/len(val_loader)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss; trigger=0
            torch.save(unet.state_dict(), 'unet_best.pth')
        else:
            trigger +=1
            if trigger>=patience:
                print('Early stopping UNet'); unet.load_state_dict(torch.load('unet_best.pth')); break
    return history

def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=1e-4, patience=10, name='model'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_acc = 0.0; trigger=0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    for epoch in range(epochs):
        model.train()
        running_loss=0.0; correct=0; total=0
        for images, labels in tqdm(train_loader, desc=f'{name} Train Epoch {epoch+1}/{epochs}'):
            images = images.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds==labels).sum().item(); total += labels.size(0)
        train_loss = running_loss / len(train_loader); train_acc = correct/total
        # validation
        model.eval()
        val_loss=0.0; correct=0; total=0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device); labels = labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds==labels).sum().item(); total += labels.size(0)
        val_loss = val_loss / len(val_loader); val_acc = correct/total
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        print(f"{name} Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc; trigger=0; torch.save(model.state_dict(), f"{name}_best.pth")
        else:
            trigger += 1
            if trigger >= patience:
                print(f'Early stopping {name}'); model.load_state_dict(torch.load(f"{name}_best.pth")); break
    return history
