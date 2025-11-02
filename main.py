import os
import pandas as pd
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from torchvision.io import decode_image
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        #print(row)
        img_name = str(row['ID']) + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        #Image.open(img_path).show()
        tensor_img = decode_image(img_path)
        labels = [col for col in self.data.columns if col not in ['ID', 'Disease_Risk']]
        labels = torch.tensor(row[labels].values.astype('int'))
        if self.transform:
            tensor_img = self.transform(tensor_img)
        if self.target_transform:
            labels = self.target_transform(labels)
        return tensor_img, labels

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 45)
        )

    def forward(self, x):
        x = self.cnn_stack(x)
        return x

def PrepData():
    transforms = v2.Compose([
        v2.Resize((224,224)),
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomImageDataset(label_file='dataset/labels.csv', img_dir='dataset', transform=transforms)
    #print(dataset.__getitem__(3))
    #print(dataset.__len__())

    train_set_size = int(0.8 * dataset.__len__())
    val_set_size = int(0.1 * dataset.__len__())
    test_set_size = dataset.__len__() - train_set_size - val_set_size

    train_data, val_data, test_data = random_split(dataset, [train_set_size, val_set_size, test_set_size])
    #print(train_data.__len__())
    #print(test_data.__len__())
    #print(val_data.__len__())

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader
    #return train_dataloader

def DisplayData(train_dataloader):
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}")
    img = train_features[0].squeeze().permute(1, 2, 0)
    label = train_labels[0]
    plt.imshow(img)
    plt.savefig("output.png")
    print(f"Labels: {label}")

def TrainLoop(dataloader, model, loss_fn, optimiser, device):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).float()
        optimiser.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()
                
        total_loss += loss.item()

        preds = torch.sigmoid(pred) > 0.5
        correct += (preds == y.bool()).sum().item()
        total += y.numel()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * 64 + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def ValidateLoop(dataloader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float()
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            
            preds = torch.sigmoid(pred) > 0.5
            correct += (preds == y.bool()).sum().item()
            total += y.numel()
    
    avg_loss = val_loss / len(dataloader)
    accuracy = correct / total
    print(f"Validation loss: {avg_loss:.6f}\n")
    return avg_loss, accuracy

def InitModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    model = CNN().to(device)
    #print(model)
    return device, model

def UseModel(device, model):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    #model = CNN().to(device)
    #print(model)

    learning_rate = 0.001
    epochs = 10
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataloader, val_dataloader, test_dataloader = PrepData()
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], [] 

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = TrainLoop(train_dataloader, model, loss_fn, optimiser, device)
        val_loss, val_acc = ValidateLoop(val_dataloader, model, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}\n")
    print("done")

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_graph.png")
    #plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy_graph.png")
    #plt.show()

device, model = InitModel()
UseModel(device, model)

