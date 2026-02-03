#TODO look into using bash scripts (https://github.com/Delphboy/SuperCap/tree/main)

import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MultilabelF1Score
from sklearn.metrics import classification_report

TRAIN_DIR = 'dataset/Training_Set/Training_Set'
TRAIN_LABELS = pd.read_csv(f'{TRAIN_DIR}/RFMiD_Training_Labels.csv')
TRAIN_LABELS.drop(columns='Disease_Risk', inplace=True)
TRAIN_DATA = f'{TRAIN_DIR}/Training'

VAL_DIR = 'dataset/Evaluation_Set/Evaluation_Set'
VAL_LABELS = pd.read_csv(f'{VAL_DIR}/RFMiD_Validation_Labels.csv')
VAL_LABELS.drop(columns='Disease_Risk', inplace=True)
VAL_DATA = f'{VAL_DIR}/Validation'

TEST_DIR = 'dataset/Test_Set/Test_Set'
TEST_LABELS = pd.read_csv(f'{TEST_DIR}/RFMiD_Testing_Labels.csv')
TEST_LABELS.drop(columns='Disease_Risk', inplace=True)
TEST_DATA = f'{TEST_DIR}/Test'

LEARNING_RATE = 1e-4
EPOCHS = 100
TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
THRESHOLD = 0.5
TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    # v2.RandomHorizontalFlip(0.5),
    # v2.RandomRotation(degrees=15),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."

class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir,transform=None, target_transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = str(row['ID']) + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        labels = torch.tensor(row.drop('ID').values)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labels = self.target_transform(labels)
        return img, labels

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 45)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

def PrepData(dataset_train, dataset_val, dataset_test):
    label_cols = [col for col in dataset_train.df.columns if col not in ['ID']]
    positives = dataset_train.df[label_cols].sum(axis=0).astype(float)
    total = len(dataset_train.df)
    negatives = total - positives
    pos_weight_vals = (negatives / (positives + 1e-5)).values
    pos_weights = torch.tensor(pos_weight_vals, dtype=torch.float32).sqrt()
    # pos_weights = torch.clamp(pos_weights, max=10.0)

    # print(dataset.__getitem__(3))
    # print(dataset.__len__())

    torch.manual_seed(42)

    train_dataloader = DataLoader(dataset_train, batch_size=TRAINING_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, pos_weights


def TrainLoop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE).float()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        total_loss += loss.item()
        preds = torch.sigmoid(pred) > THRESHOLD
        correct += (preds == y.bool()).sum().item()
        total += y.numel()

        loss, current = loss.item(), batch * TRAINING_BATCH_SIZE + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def ValLoop(dataloader, model, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    total_positive_preds = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE).float()
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            preds = torch.sigmoid(pred) > THRESHOLD
            total_positive_preds += preds.sum().item()
            correct += (preds == y.bool()).sum().item()
            total += y.numel()
    print(f"DEBUG: Total positive predictions in Validation: {total_positive_preds}")
    avg_loss = val_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def TestModel(model, loss_fn, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    metric = MultilabelF1Score(num_labels=45, average='micro', threshold=0.5).to(DEVICE)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE).float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            preds = torch.sigmoid(pred) > THRESHOLD
            correct += (preds == y.bool()).sum().item()
            total += y.numel()
            metric.update(pred, y)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    avg_loss = test_loss / len(dataloader)
    accuracy = correct / total
    f1_score = metric.compute().item()

    val_preds_np = np.vstack(all_preds)
    val_targets_np = np.vstack(all_targets)
    print(classification_report(val_targets_np, val_preds_np, zero_division=0))

    return avg_loss, accuracy, f1_score


def UseModel(model, dataset_train, dataset_val, dataset_test):
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.7, patience=3)

    train_dataloader, val_dataloader, test_dataloader, pos_weights = PrepData(dataset_train, dataset_val, dataset_test)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(DEVICE))

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = TrainLoop(train_dataloader, model, loss_fn, optimiser)
        val_loss, val_acc = ValLoop(val_dataloader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}\n")
    print("done")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_graph.png")
    # plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy_graph.png")
    # plt.show()

    test_loss, test_acc, f1_score = TestModel(model, loss_fn, test_dataloader)
    print(f'test loss = {test_loss:.4f}')
    print(f'test acc = {test_acc:.4f}')
    print(f'f1 score = {f1_score:.4f}')


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
    dataset_train = CustomImageDataset(df=TRAIN_LABELS, img_dir=TRAIN_DATA, transform=TRANSFORMS)
    dataset_val = CustomImageDataset(df=VAL_LABELS, img_dir=VAL_DATA, transform=TRANSFORMS)
    dataset_test = CustomImageDataset(df=TEST_LABELS, img_dir=TEST_DATA, transform=TRANSFORMS)
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(DEVICE)
    UseModel(model, dataset_train, dataset_val, dataset_test)