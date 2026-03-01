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
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from sklearn.metrics import classification_report
from datetime import datetime
from torchinfo import summary

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

RES_BLOCKS = [3, 4, 6, 3]
LEARNING_RATE = 1e-4
USE_WEIGHT_BIAS = True
WEIGHT_DECAY = 1e-3
EPOCHS = 150
THRESHOLD = 0.5
TRAINING_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
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

class CreatePatches(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TwoLayerNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layer(x) + x


class Block(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features*4)
        self.out_layer2 = TwoLayerNN(in_features, in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features*2, in_features, 1, 1, groups=head_num)

    def forward(self, x):
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation
        neibor_features = x[torch.arange(
            B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack(
            [x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(self.out_layer1(
            nn.functional.gelu(x).view(B * N, -1)).view(B, N, -1))
        x = x + shortcut

        x = self.droppath2(self.out_layer2(nn.functional.gelu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1)) + x

        return x

class GNN(nn.Module):
    def __init__(self, in_features=3*16*16, out_feature=320, num_patches=196, num_ViGBlocks=16, num_edges=9, head_num=1):
        super().__init__()

        self.patchifier = CreatePatches()
        # self.patch_embedding = TwoLayerNN(in_features)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//8),
            nn.BatchNorm1d(out_feature//8),
            nn.GELU(),
            nn.Linear(out_feature//8, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature),
            nn.BatchNorm1d(out_feature)
        )
        self.pose_embedding = nn.Parameter(
            torch.rand(num_patches, out_feature))

        self.blocks = nn.Sequential(
            *[Block(out_feature, num_edges, head_num)
              for _ in range(num_ViGBlocks)])

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C = x.shape
        x = self.patch_embedding(x.reshape(B * N, -1)).reshape(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x


class Classifier(nn.Module):
    def __init__(self, in_features=3*16*16, out_feature=320, num_patches=196, num_ViGBlocks=16, hidden_layer=1024, num_edges=9, head_num=1, n_classes=1):
        super().__init__()
        self.backbone = GNN(in_features, out_feature,
                             num_patches, num_ViGBlocks,
                             num_edges, head_num)

        self.predictor = nn.Sequential(
            nn.Linear(out_feature*num_patches, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Linear(hidden_layer, n_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        B, N, C = features.shape
        x = self.predictor(features.view(B, -1))
        return features, x

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
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
    # pos_weights = torch.clamp(pos_weights, max=10.0)

    # print(dataset.__getitem__(3))
    # print(dataset.__len__())

    torch.manual_seed(42)

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
        _, outputs = model(inputs)
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
            _, outputs = model(inputs)
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
    # loss_fn = nn.CrossEntropyLoss()

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
                _, outputs = model(inputs)
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
    summary_path = 'vit.txt'
    header = "date;time;learning_rate;weight_decay;weight_parameter;Threshold;epochs;early_stopping;train_transforms;test_transforms;precision;recall;f1_score\n"

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    label_cols = [c for c in dataset_train.df.columns if c != 'ID']
    classes_str = ",".join(label_cols)

    weight_param_used = True if USE_WEIGHT_BIAS else False
    early_stopping_used = True if early_stopping.early_stop else False

    line = (
        f"{date_str};"
        f"{time_str};"
        f"{RES_BLOCKS};"
        f"{LEARNING_RATE};"
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
    dataset_train = CustomImageDataset(df=TRAIN_LABELS, img_dir=TRAIN_DATA, transform=TRAIN_TRANSFORMS)
    dataset_val = CustomImageDataset(df=VAL_LABELS, img_dir=VAL_DATA, transform=TEST_TRANSFORMS)
    dataset_test = CustomImageDataset(df=TEST_LABELS, img_dir=TEST_DATA, transform=TEST_TRANSFORMS)
    model = Classifier().to(DEVICE)
    print(summary(model, input_size=(TRAINING_BATCH_SIZE, 3, 224, 224)))
    UseModel(model, dataset_train, dataset_val, dataset_test)
