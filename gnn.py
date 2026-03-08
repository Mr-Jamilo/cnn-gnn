#TODO look into using bash scripts (https://github.com/Delphboy/SuperCap/tree/main)

import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from sklearn.metrics import classification_report
from datetime import datetime
from torchinfo import summary
from timm.layers.drop import DropPath
from gcn_lib.torch_vertex import DyGraphConv2d

TRAIN_DIR = 'dataset/Training_Set/Training_Set'
TRAIN_LABELS = pd.read_csv(f'{TRAIN_DIR}/RFMiD_Training_Labels.csv')
TRAIN_LABELS = TRAIN_LABELS[['ID', 'Disease_Risk']]
TRAIN_DATA = f'{TRAIN_DIR}/Training'

VAL_DIR = 'dataset/Evaluation_Set/Evaluation_Set'
VAL_LABELS = pd.read_csv(f'{VAL_DIR}/RFMiD_Validation_Labels.csv')
VAL_LABELS = VAL_LABELS[['ID', 'Disease_Risk']]
VAL_DATA = f'{VAL_DIR}/Validation'

TEST_DIR = 'dataset/Test_Set/Test_Set'
TEST_LABELS = pd.read_csv(f'{TEST_DIR}/RFMiD_Testing_Labels.csv')
TEST_LABELS = TEST_LABELS[['ID', 'Disease_Risk']]
TEST_DATA = f'{TEST_DIR}/Test'

# pyramidvig
DEPTH = [2, 2, 6, 2] 
CHANNELS = [80, 160, 400, 640]
STOCHASTIC_PATH = 0.1
K_NEIGHBOURS = 7
LEARNING_RATE = 1e-4
USE_WEIGHT_BIAS = True
WEIGHT_DECAY = 1e-3
EPOCHS = 150
THRESHOLD = 0.5
TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
train_transform_list = [v2.ToImage(),v2.Resize((224, 224)), v2.RandomHorizontalFlip(0.5), v2.RandomRotation(degrees=15), v2.ConvertImageDtype(torch.float), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
TRAIN_TRANSFORMS = v2.Compose(train_transform_list)
test_transform_list = [v2.ToImage(), v2.Resize((224, 224)), v2.ConvertImageDtype(torch.float), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
TEST_TRANSFORMS = v2.Compose(test_transform_list)

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
        labels = torch.tensor(row['Disease_Risk'])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labels = self.target_transform(labels)
        return img, labels

class Stem(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class GrapherModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, k, dilation, drop_path=0.0):
        super(GrapherModule, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = nn.Sequential(
            DyGraphConv2d(in_channels, hidden_channels, k, dilation, act="None"),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class FFNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, drop_path=0.0):
        super(FFNModule, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class ViGBlock(nn.Module):
    def __init__(self, channels, k, dilation, drop_path=0.0):
        super(ViGBlock, self).__init__()
        self.grapher = GrapherModule(channels, channels*2, k, dilation, drop_path)
        self.fnn = FFNModule(channels, channels*4, drop_path)

    def forward(self, x):
        x = self.grapher(x)
        x = self.fnn(x)
        return x

class ViGNN(nn.Module):
    def __init__(self, in_channels, num_classes, k, depths, channels, drop_path):
        super(ViGNN, self).__init__()
        self.num_classes = num_classes
        self.stem = Stem(in_dim=in_channels, out_dim=channels[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                ViGBlock(channels[i], k, dilation=1, drop_path=drop_path)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
            if i < len(depths) - 1:
                self.downsamples.append(Downsample(in_dim=channels[i], out_dim=channels[i+1]))
        self.norm = nn.BatchNorm1d(channels[-1])
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i < len(self.stages) - 1:
                x = x.reshape(B, -1, H, W)
                x = self.downsamples[i](x)
                B, C, H, W = x.shape
        x = x.flatten(2).mean(dim=2)
        x = self.norm(x)
        x = self.head(x)
        return x

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def PrepData(dataset_train, dataset_val, dataset_test):
    label_cols = [col for col in dataset_train.df.columns if col not in ['ID']]
    positives = dataset_train.df[label_cols].sum(axis=0).astype(float)
    total = len(dataset_train.df)
    negatives = total - positives
    pos_weight_vals = (negatives / positives).values
    pos_weights = torch.tensor(pos_weight_vals, dtype=torch.float32)

    # print(dataset.__getitem__(3))
    # print(dataset.__len__())

    train_dataloader = DataLoader(dataset_train, batch_size=TRAINING_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, pos_weights

def train_one_epoch(dataloader, model, loss_fn, optimiser):
    loss_list = []
    correct = 0
    total = 0
    f1 = BinaryF1Score().to(DEVICE)

    for batch, data in enumerate(dataloader):
        inputs, targets = data
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimiser.step()
        loss_list.append(loss.item())
        outputs = torch.sigmoid(outputs) > THRESHOLD
        f1.update(outputs, targets)

        correct += (outputs == targets.bool()).sum().item()
        total += targets.numel()

        print(f"  Batch {batch + 1}/{len(dataloader)} - Loss: {loss.item():.4f} - Acc: {(correct/total):.4f}", end='\r')

    avg_loss = np.mean(loss_list)
    avg_accuracy = correct / total
    f1 = f1.compute().item()
    return avg_loss, avg_accuracy, f1

def TestModel(model, loss_fn, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    metric = BinaryF1Score().to(DEVICE)
    precision_metric = BinaryPrecision().to(DEVICE)
    recall_metric = BinaryRecall().to(DEVICE)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(inputs)
            test_loss += loss_fn(outputs, labels).item()
            outputs = torch.sigmoid(outputs) > THRESHOLD
            correct += (outputs == labels.bool()).sum().item()
            total += labels.numel()
            metric.update(outputs, labels)
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    avg_loss = test_loss / len(dataloader)
    accuracy = correct / total
    f1_score = metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()

    val_preds_np = np.vstack(all_preds)
    val_targets_np = np.vstack(all_targets)
    print(classification_report(val_targets_np, val_preds_np, zero_division=0))

    return avg_loss, accuracy, f1_score, precision, recall

def UseModel(model, dataset_train, dataset_val, dataset_test):
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.7, patience=10)
    early_stopping = EarlyStopping(patience=20, delta=0.00001)

    train_dataloader, val_dataloader, test_dataloader, pos_weights = PrepData(dataset_train, dataset_val, dataset_test)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(DEVICE))

    actual_epochs = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train(True)
        train_loss, train_acc, train_f1 = train_one_epoch(train_dataloader, model, loss_fn, optimiser)
        # train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimiser)
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        val_f1 = BinaryF1Score().to(DEVICE)
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(inputs)
                running_val_loss += loss_fn(outputs, labels).item()

                outputs = torch.sigmoid(outputs) > THRESHOLD
                val_f1.update(outputs, labels)
                correct += (outputs == labels.bool()).sum().item()
                total += labels.numel()

        val_loss = running_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0
        val_acc = correct / total
        val_f1 = val_f1.compute().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        scheduler.step(val_loss)

        early_stopping(val_loss, model)
        actual_epochs += 1
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}\n")
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

    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.title('F1 over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig("f1_graph.png")
    # plt.show()

    early_stopping.load_best_model(model)
    test_loss, test_acc, f1_score, precision, recall = TestModel(model, loss_fn, test_dataloader)
    print(f'test loss = {test_loss:.4f}')
    print(f'test acc = {test_acc:.4f}')
    print(f'test f1 score = {f1_score:.4f}')

    # Logging
    summary_path = 'gnn.txt'
    header = "date;time;learning_rate;k-neighbours;channels;depth;stochastic_path;weight_decay;weight_parameter;Threshold;epochs;early_stopping;train_transforms;test_transforms;precision;recall;f1_score\n"

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    weight_param_used = True if USE_WEIGHT_BIAS else False
    early_stopping_used = True if early_stopping.early_stop else False

    line = (
        f"{date_str};"
        f"{time_str};"
        f"{LEARNING_RATE};"
        f"{K_NEIGHBOURS};"
        f"{CHANNELS};"
        f"{DEPTH};"
        f"{STOCHASTIC_PATH};"
        f"{WEIGHT_DECAY};"
        f"{str(weight_param_used)};"
        f"{THRESHOLD};"
        f"{actual_epochs};"
        f"{str(early_stopping_used)};"
        f"{str(train_transform_list)};"
        f"{str(test_transform_list)};"
        f"{precision:.4f};"
        f"{recall:.4f};"
        f"{f1_score:.4f}\n"
    )

    write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(line)

if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
    torch_geometric.seed_everything(42)
    dataset_train = CustomImageDataset(df=TRAIN_LABELS, img_dir=TRAIN_DATA, transform=TRAIN_TRANSFORMS)
    dataset_val = CustomImageDataset(df=VAL_LABELS, img_dir=VAL_DATA, transform=TEST_TRANSFORMS)
    dataset_test = CustomImageDataset(df=TEST_LABELS, img_dir=TEST_DATA, transform=TEST_TRANSFORMS)
    model = ViGNN(in_channels=3, num_classes=1, k=K_NEIGHBOURS, depths=DEPTH, channels=CHANNELS, drop_path=STOCHASTIC_PATH).to(DEVICE)
    print(summary(model, input_size=(TRAINING_BATCH_SIZE, 3, 224, 224)))
    UseModel(model, dataset_train, dataset_val, dataset_test)
