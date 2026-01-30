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
TEST_DIR = 'dataset/Test_Set/Test_Set'
EVAL_DIR = 'dataset/Evaluation_Set/Evaluation_Set'
TRAIN_LABELS = pd.read_csv(f'{TRAIN_DIR}/RFMiD_Training_Labels.csv')
TEST_LABELS = pd.read_csv(f'{TEST_DIR}/RFMiD_Testing_Labels.csv')
EVAL_LABELS = pd.read_csv(f'{EVAL_DIR}/RFMiD_Validation_Labels.csv')
TRAIN_DATA = f'{TRAIN_DIR}/Training'
TEST_DATA = f'{TEST_DIR}/Test'
EVAL_DATA = f'{EVAL_DIR}/Validation'

LEARNING_RATE = 3e-4
EPOCHS = 200
MINIMUM_CLASS_EXAMPLES = 150
TRAINING_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
THRESHOLD = 0.3
TRANSFORMS = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."

class CustomImageDataset(Dataset):
    def __init__(self, df, valid_labels, transform=None, target_transform=None):
        self.labels_df = df[['ID', 'Disease_Risk'] + valid_labels]
        self.classes_count = len(valid_labels)
        self.valid_labels = valid_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        img_name = str(row['ID']) + ".png"
        img_path = os.path.join(TRAIN_DATA, img_name)
        img = Image.open(img_path).convert('RGB')
        labels = [col for col in self.labels_df.columns if col not in ['ID', 'Disease_Risk']]
        labels = torch.tensor(row[labels].values.astype('int'))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labels = self.target_transform(labels)
        return img, labels


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

def dropClasses():
    label_cols = [col for col in TRAIN_LABELS.columns if col not in ['ID', 'Disease_Risk']]
    class_counts = TRAIN_LABELS[label_cols].sum(axis=0)
    valid_labels = class_counts[class_counts >= MINIMUM_CLASS_EXAMPLES].index.tolist()

    print(f"Original classes: {len(label_cols)}")
    print(f"Classes with >= {MINIMUM_CLASS_EXAMPLES} examples: {len(valid_labels)}")
    print(f"Dropped: {set(label_cols) - set(valid_labels)}")

    return valid_labels

def PrepData(dataset_train, dataset_test, dataset_eval):
    label_cols = [col for col in dataset_train.labels_df.columns if col not in ['ID', 'Disease_Risk']]
    positives = dataset_train.labels_df[label_cols].sum(axis=0).astype(float)
    total = len(dataset_train.labels_df)
    negatives = total - positives
    pos_weight_vals = (negatives / (positives + 1e-5)).values
    pos_weights = torch.tensor(pos_weight_vals, dtype=torch.float32).sqrt()
    # pos_weights = torch.clamp(pos_weights, max=10.0)

    # print(dataset.__getitem__(3))
    # print(dataset.__len__())

    torch.manual_seed(42)

    train_dataloader = DataLoader(dataset_train, batch_size=TRAINING_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    eval_dataloader = DataLoader(dataset_eval, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, test_dataloader, eval_dataloader, pos_weights


def DisplayData(train_dataloader):
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}")
    img = train_features[0].squeeze().permute(1, 2, 0)
    label = train_labels[0]
    plt.imshow(img)
    plt.savefig("output.png")
    print(f"Labels: {label}")


def TrainLoop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):
        # print("DEBUG", f"{y[0]}")
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


def TestLoop(dataloader, model, loss_fn):
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


def EvalModel(model, loss_fn, dataloader, num_classes):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    metric = MultilabelF1Score(num_labels=num_classes, average='micro', threshold=THRESHOLD).to(DEVICE)

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


def UseModel(model, dataset_train, dataset_test, dataset_eval, num_classes):
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.7, patience=3)

    train_dataloader, test_dataloader, eval_dataloader, pos_weights = PrepData(dataset_train, dataset_test, dataset_eval)
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(DEVICE))
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = TrainLoop(train_dataloader, model, loss_fn, optimiser)
        test_loss, test_acc = TestLoop(test_dataloader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(test_acc)
        # scheduler.step(test_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss:   {test_loss:.4f}, Test Acc:   {test_acc:.4f}\n")
    print("done")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_graph.png")
    # plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy_graph.png")
    # plt.show()

    eval_loss, eval_acc, f1_score = EvalModel(model, loss_fn, eval_dataloader, num_classes)
    print(f'eval loss = {eval_loss:.4f}')
    print(f'eval acc = {eval_acc:.4f}')
    print(f'eval f1 score = {f1_score:.4f}')


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
    valid_labels = dropClasses()
    dataset_train = CustomImageDataset(df=TRAIN_LABELS, valid_labels=valid_labels, transform=TRANSFORMS)
    dataset_test = CustomImageDataset(df=TEST_LABELS, valid_labels=valid_labels, transform=TRANSFORMS)
    dataset_eval = CustomImageDataset(df=EVAL_LABELS, valid_labels=valid_labels, transform=TRANSFORMS)
    model = LeNet(num_classes=dataset_train.classes_count).to(DEVICE)
    UseModel(model, dataset_train, dataset_test, dataset_eval, dataset_train.classes_count)