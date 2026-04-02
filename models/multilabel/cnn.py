import os
import opts
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall
from sklearn.metrics import classification_report
from datetime import datetime
from torchinfo import summary

NUM_CLASSES = 4
LABEL_NAMES = ["DR", "MH", "TSLN", "ODC"]

train_transform_list = [v2.ToImage(), v2.Resize((224, 224)), v2.RandomHorizontalFlip(0.5), v2.RandomRotation(degrees=15), v2.ConvertImageDtype(torch.float), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
TRAIN_TRANSFORMS = v2.Compose(train_transform_list)
test_transform_list = [v2.ToImage(), v2.Resize((224, 224)), v2.ConvertImageDtype(torch.float), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
TEST_TRANSFORMS = v2.Compose(test_transform_list)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), ("CUDA is not available. Please run on a machine with a GPU.")


class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, target_transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = str(row["ID"]) + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        labels = torch.tensor(row.drop("ID").values, dtype=torch.float32)
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
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, num_classes)

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
        out = self.dropout(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

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
        os.makedirs("../../weights/multilabel", exist_ok=True)
        torch.save(self.best_model_state, "../../weights/multilabel/cnn.pth")
        model.load_state_dict(self.best_model_state)

def PrepData(opt, dataset_train, dataset_val, dataset_test):
    label_cols = [col for col in dataset_train.df.columns if col not in ["ID"]]
    positives = dataset_train.df[label_cols].sum(axis=0).astype(float)
    total = len(dataset_train.df)
    negatives = total - positives
    pos_weight_vals = (negatives / positives).values
    pos_weights = torch.tensor(pos_weight_vals, dtype=torch.float32)

    if opt.seed != -1:
        torch.manual_seed(opt.seed)

    train_dataloader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, pos_weights

def train_one_epoch(opt, dataloader, model, loss_fn, optimiser):
    loss_list = []
    correct = 0
    total = 0
    f1 = MultilabelF1Score(num_labels=NUM_CLASSES).to(DEVICE)

    for batch, data in enumerate(dataloader):
        inputs, targets = data
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).float()
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimiser.step()
        loss_list.append(loss.item())
        outputs = torch.sigmoid(outputs) > opt.threshold
        f1.update(outputs, targets)

        correct += (outputs == targets.bool()).sum().item()
        total += targets.numel()

        print(f"  Batch {batch + 1}/{len(dataloader)} - Loss: {loss.item():.4f} - Acc: {(correct / total):.4f}", end="\r")

    avg_loss = np.mean(loss_list)
    avg_accuracy = correct / total
    f1 = f1.compute().item()
    return avg_loss, avg_accuracy, f1

def TestModel(opt, model, loss_fn, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    metric = MultilabelF1Score(num_labels=NUM_CLASSES).to(DEVICE)
    precision_metric = MultilabelPrecision(num_labels=NUM_CLASSES).to(DEVICE)
    recall_metric = MultilabelRecall(num_labels=NUM_CLASSES).to(DEVICE)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
            outputs = model(inputs)
            test_loss += loss_fn(outputs, labels).item()
            outputs = torch.sigmoid(outputs) > opt.threshold
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
    print(classification_report(val_targets_np, val_preds_np, target_names=LABEL_NAMES, zero_division=0))

    return avg_loss, accuracy, f1_score, precision, recall

def UseModel(opt, model, dataset_train, dataset_val, dataset_test):
    if opt.weight_decay != -1:
        optimiser = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.7, patience=10)
    early_stopping = EarlyStopping(patience=25, delta=0.0001)

    train_dataloader, val_dataloader, test_dataloader, pos_weights = PrepData(opt, dataset_train, dataset_val, dataset_test)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(DEVICE))

    actual_epochs = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch + 1}/{opt.epochs}")
        model.train(True)
        train_loss, train_acc, train_f1 = train_one_epoch(opt, train_dataloader, model, loss_fn, optimiser)
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        val_f1 = MultilabelF1Score(num_labels=NUM_CLASSES).to(DEVICE)

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
                outputs = model(inputs)
                running_val_loss += loss_fn(outputs, labels).item()

                preds = torch.sigmoid(outputs) > opt.threshold
                val_f1.update(preds, labels)
                correct += (preds == labels.bool()).sum().item()
                total += labels.numel()

        val_loss = (running_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0)
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
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("cnn_loss_graph.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("cnn_accuracy_graph.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label="Train F1")
    plt.plot(val_f1s, label="Val F1")
    plt.title("F1 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.savefig("cnn_f1_graph.png")

    early_stopping.load_best_model(model)
    test_loss, test_acc, f1_score, precision, recall = TestModel(opt, model, loss_fn, test_dataloader)
    print(f"test loss = {test_loss:.4f}")
    print(f"test acc = {test_acc:.4f}")
    print(f"test f1 score = {f1_score:.4f}")

    summary_path = "cnn.txt"
    header = "date;time;res_blocks;classes;learning_rate;weight_decay;weight_parameter;threshold;epochs;early_stopping;train_transforms;test_transforms;precision;recall;f1_score\n"

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    label_cols = [c for c in dataset_train.df.columns if c != "ID"]
    classes_str = ",".join(label_cols)

    weight_param_used = opt.weight_decay != -1
    early_stopping_used = early_stopping.early_stop

    line = (
        f"{date_str};"
        f"{time_str};"
        f"{opt.cnn_res_blocks_list};"
        f"{len(label_cols)}({classes_str});"
        f"{opt.learning_rate};"
        f"{opt.weight_decay};"
        f"{str(weight_param_used)};"
        f"{opt.threshold};"
        f"{actual_epochs};"
        f"{str(early_stopping_used)};"
        f"{str(train_transform_list)};"
        f"{str(test_transform_list)};"
        f"{precision:.4f};"
        f"{recall:.4f};"
        f"{f1_score:.4f}\n"
    )

    write_header = (not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0)
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(line)

def predict_image(model, image_path, opt):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = TEST_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.sigmoid(output).squeeze(0).cpu().numpy()
        predictions = (probabilities > opt.threshold).astype(int)

    pred_dict = {name: int(pred) for name, pred in zip(LABEL_NAMES, predictions)}
    prob_dict = {
        name: round(float(prob), 4) for name, prob in zip(LABEL_NAMES, probabilities)
    }
    active_diseases = [name for name, pred in pred_dict.items() if pred == 1]

    return {
        "predictions": pred_dict,
        "probabilities": prob_dict,
        "active_diseases": active_diseases,
    }


if __name__ == "__main__":
    opt = opts.parse_opts()
    assert torch.cuda.is_available(), ("CUDA is not available. Please run on a machine with a GPU.")

    if opt.seed != -1:
        torch.manual_seed(opt.seed)

    dataset_directory = opt.dataset_directory

    TRAIN_DIR = f"{dataset_directory}/Training_Set/Training_Set"
    TRAIN_LABELS = pd.read_csv(f"{TRAIN_DIR}/RFMiD_Training_Labels.csv")
    TRAIN_LABELS = TRAIN_LABELS[["ID", "DR", "MH", "TSLN", "ODC"]]
    TRAIN_DATA = f"{TRAIN_DIR}/Training"

    VAL_DIR = f"{dataset_directory}/Evaluation_Set/Evaluation_Set"
    VAL_LABELS = pd.read_csv(f"{VAL_DIR}/RFMiD_Validation_Labels.csv")
    VAL_LABELS = VAL_LABELS[["ID", "DR", "MH", "TSLN", "ODC"]]
    VAL_DATA = f"{VAL_DIR}/Validation"

    TEST_DIR = f"{dataset_directory}/Test_Set/Test_Set"
    TEST_LABELS = pd.read_csv(f"{TEST_DIR}/RFMiD_Testing_Labels.csv")
    TEST_LABELS = TEST_LABELS[["ID", "DR", "MH", "TSLN", "ODC"]]
    TEST_DATA = f"{TEST_DIR}/Test"

    dataset_train = CustomImageDataset(df=TRAIN_LABELS, img_dir=TRAIN_DATA, transform=TRAIN_TRANSFORMS)
    dataset_val = CustomImageDataset(df=VAL_LABELS, img_dir=VAL_DATA, transform=TEST_TRANSFORMS)
    dataset_test = CustomImageDataset(df=TEST_LABELS, img_dir=TEST_DATA, transform=TEST_TRANSFORMS)

    model = ResNet(ResidualBlock, opt.cnn_res_blocks_list, NUM_CLASSES).to(DEVICE)
    print(summary(model, input_size=(opt.batch_size, 3, 224, 224)))
    UseModel(opt, model, dataset_train, dataset_val, dataset_test)
