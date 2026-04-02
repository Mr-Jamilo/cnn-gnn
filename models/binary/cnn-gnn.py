import os
import opts
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric import nn as tg_nn
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import F1Score, Precision, Recall
from sklearn.metrics import classification_report
from datetime import datetime
from torchinfo import summary
from timm.layers.drop import DropPath
from gcn_lib.torch_vertex import DyGraphConv2d
from torch_cluster import knn_graph

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
        labels = torch.tensor(row["Disease_Risk"])
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
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CNNFeatureExtractor(nn.Module):
    def __init__(self, block, num_blocks, extraction_layer="layer4"):
        super(CNNFeatureExtractor, self).__init__()
        self.in_channels = 64
        self.extraction_layer = extraction_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        if extraction_layer in ["layer4", "avgpool"]:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if extraction_layer == "avgpool":
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_channels = {
            "layer3": 256,
            "layer4": 512,
            "avgpool": 512,
        }[extraction_layer]

        self.out_spatial = {
            "layer3": 14,
            "layer4": 7,
            "avgpool": 1,
        }[extraction_layer]

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.extraction_layer == "layer3":
            return x

        x = self.layer4(x)

        if self.extraction_layer == "layer4":
            return x

        x = self.avgpool(x)
        return x

class DyGraphAtt2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, heads=1):
        super(DyGraphAtt2d, self).__init__()
        self.k = k
        self.gat = tg_nn.GATConv(in_channels, out_channels, heads=heads, concat=False)

    def forward(self, x):
        B, C, N, _ = x.shape
        x_flat = x.squeeze(-1).transpose(1, 2).reshape(B * N, C)
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
        edge_index = knn_graph(x_flat, self.k, batch_idx, loop=True)
        out = self.gat(x_flat, edge_index)
        out = out.reshape(B, N, -1).transpose(1, 2).unsqueeze(-1)
        return out

class DyGraphGIN2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, eps=0.0, train_eps=True):
        super(DyGraphGIN2d, self).__init__()
        self.k = k
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        self.gin = tg_nn.GINConv(mlp, eps=eps, train_eps=train_eps)

    def forward(self, x):
        B, C, N, _ = x.shape
        x_flat = x.squeeze(-1).transpose(1, 2).reshape(B * N, C)
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
        edge_index = knn_graph(x_flat, self.k, batch_idx, loop=True)
        out = self.gin(x_flat, edge_index)
        out = out.reshape(B, N, -1).transpose(1, 2).unsqueeze(-1)
        return out

class GrapherModule(nn.Module):
    def __init__(self, opt, in_channels, hidden_channels, k, dilation, drop_path=0.0):
        super(GrapherModule, self).__init__()
        self.graph_type = opt.graph_layer_type
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )

        self.graph_conv = nn.Sequential(
            DyGraphConv2d(in_channels, hidden_channels, k, dilation, act="None"),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )

        self.graph_att = nn.Sequential(
            DyGraphAtt2d(in_channels, hidden_channels, k=k, heads=4),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )

        self.graph_iso = nn.Sequential(
            DyGraphGIN2d(in_channels, hidden_channels, k=k),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        shortcut = x
        x = self.fc1(x)
        match self.graph_type:
            case "GCN":
                x = self.graph_conv(x)
            case "GAT":
                x = self.graph_att(x)
            case "GIN":
                x = self.graph_iso(x)
            case _:
                raise ValueError(f"Unrecognised graph type: {self.graph_type}")
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x.reshape(B, C, H, W)

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
            nn.BatchNorm2d(in_channels)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class ViGBlock(nn.Module):
    def __init__(self, opt, channels, k, dilation, drop_path=0.0):
        super(ViGBlock, self).__init__()
        self.grapher = GrapherModule(opt, channels, channels * 2, k, dilation, drop_path)
        self.ffn = FFNModule(channels, channels * 4, drop_path)

    def forward(self, x):
        x = self.grapher(x)
        x = self.ffn(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        return self.conv(x)

class GNNClassifier(nn.Module):
    def __init__(self, opt, in_channels, in_spatial, num_classes, channels, k, drop_path):
        super(GNNClassifier, self).__init__()

        self.num_classes = num_classes
        self.in_spatial = in_spatial

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(channels)):
            stage = nn.Sequential(
                ViGBlock(opt, channels[i], k, dilation=1, drop_path=drop_path),
                ViGBlock(opt, channels[i], k, dilation=1, drop_path=drop_path),
            )
            self.stages.append(stage)

            if i < len(channels) - 1:
                self.downsamples.append(Downsample(channels[i], channels[i + 1]))

        self.norm = nn.BatchNorm1d(channels[-1])
        self.head = nn.Linear(channels[-1], num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.projection(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x)
        x = x.flatten(2).mean(dim=2)
        x = self.norm(x)
        x = self.head(x)

        return x

class CNNGNNModel(nn.Module):
    def __init__(self, opt, cnn_blocks, extraction_layer, gnn_channels, k, drop_path):
        super(CNNGNNModel, self).__init__()

        self.cnn = CNNFeatureExtractor(ResidualBlock, cnn_blocks, extraction_layer)
        self.gnn = GNNClassifier(opt=opt, in_channels=self.cnn.out_channels, in_spatial=self.cnn.out_spatial, num_classes=1, channels=gnn_channels, k=k, drop_path=drop_path)

        self.extraction_layer = extraction_layer
        self.cnn_out_channels = self.cnn.out_channels
        self.cnn_out_spatial = self.cnn.out_spatial

    def forward(self, x):
        features = self.cnn(x)
        output = self.gnn(features)
        return output

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
        os.makedirs("../../weights/binary", exist_ok=True)
        torch.save(self.best_model_state, "../../weights/binary/cnn-gnn.pth")
        model.load_state_dict(self.best_model_state)

def PrepData(opt, dataset_train, dataset_val, dataset_test):
    label_cols = [col for col in dataset_train.df.columns if col not in ["ID"]]
    positives = dataset_train.df[label_cols].sum(axis=0).astype(float)
    total = len(dataset_train.df)
    negatives = total - positives
    pos_weight_vals = (negatives / positives).values
    dampened_weights = np.sqrt(pos_weight_vals)
    pos_weights = torch.tensor(pos_weight_vals, dtype=torch.float32)

    train_dataloader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, pos_weights

def train_one_epoch(opt, dataloader, model, loss_fn, optimiser):
    loss_list = []
    correct = 0
    total = 0
    f1 = F1Score(task="multiclass", num_classes=2, average="macro").to(DEVICE)

    for batch, data in enumerate(dataloader):
        inputs, targets = data
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)
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
    metric = f1score(task="multiclass", num_classes=2, average="macro").to(device)
    precision_metric = precision(task="multiclass", num_classes=2, average="macro").to(device)
    recall_metric = recall(task="multiclass", num_classes=2, average="macro").to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
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
    print(classification_report(val_targets_np, val_preds_np, zero_division=0))

    return avg_loss, accuracy, f1_score, precision, recall


def UseModel(opt, model, dataset_train, dataset_val, dataset_test, gnn_channels):
    if opt.weight_decay != -1:
        optimiser = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.7, patience=10)
    early_stopping = EarlyStopping(patience=20, delta=0.00001)

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
        val_f1 = F1Score(task="multiclass", num_classes=2, average="macro").to(DEVICE)

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = (inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1))
                outputs = model(inputs)
                running_val_loss += loss_fn(outputs, labels).item()

                outputs = torch.sigmoid(outputs) > opt.threshold
                val_f1.update(outputs, labels)
                correct += (outputs == labels.bool()).sum().item()
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
    plt.savefig("cnn_gnn_loss_graph.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("cnn_gnn_accuracy_graph.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label="Train F1")
    plt.plot(val_f1s, label="Val F1")
    plt.title("F1 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.savefig("cnn_gnn_f1_graph.png")

    early_stopping.load_best_model(model)
    test_loss, test_acc, f1_score, precision, recall = TestModel(opt, model, loss_fn, test_dataloader)
    print(f"test loss = {test_loss:.4f}")
    print(f"test acc = {test_acc:.4f}")
    print(f"test f1 score = {f1_score:.4f}")

    summary_path = "cnn-gnn.txt"
    header = ("date;time;learning_rate;k-neighbours;gnn_channels;graph_layer_type;stochastic_path;""cnn_extraction_layer;cnn_res_blocks;weight_decay;weight_parameter;threshold;epochs;""early_stopping;train_transforms;test_transforms;precision;recall;f1_score\n")

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    weight_param_used = True if opt.weight_decay != -1 else False
    early_stopping_used = True if early_stopping.early_stop else False

    line = (
        f"{date_str};"
        f"{time_str};"
        f"{opt.learning_rate};"
        f"{opt.k_neighbours};"
        f"{gnn_channels};"
        f"{opt.graph_layer_type};"
        f"{opt.stochastic_path};"
        f"{opt.cnn_extraction_layer};"
        f"{opt.cnn_res_blocks};"
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
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > opt.threshold else 0

    return {"disease_risk": prediction, "probability": round(probability, 4)}

if __name__ == "__main__":
    opt = opts.parse_opts()
    assert torch.cuda.is_available(), ("CUDA is not available. Please run on a machine with a GPU.")

    if opt.seed != -1:
        torch_geometric.seed_everything(opt.seed)

    extraction_channels = {"layer3": 256, "layer4": 512, "avgpool": 512}
    cnn_out_ch = extraction_channels[opt.cnn_extraction_layer]

    match opt.size:
        case "tiny":
            gnn_channels = [cnn_out_ch // 4, cnn_out_ch // 2]
        case "small":
            gnn_channels = [cnn_out_ch // 2, cnn_out_ch]
        case "medium":
            gnn_channels = [cnn_out_ch, cnn_out_ch * 3 // 2]
        case "big":
            gnn_channels = [cnn_out_ch, cnn_out_ch * 2]
        case _:
            raise ValueError(f"Unknown model size: {opt.size}. Must be tiny|small|medium|big")

    dataset_directory = opt.dataset_directory

    TRAIN_DIR = f"{dataset_directory}/Training_Set/Training_Set"
    TRAIN_LABELS = pd.read_csv(f"{TRAIN_DIR}/RFMiD_Training_Labels.csv")
    TRAIN_LABELS = TRAIN_LABELS[["ID", "Disease_Risk"]]
    TRAIN_DATA = f"{TRAIN_DIR}/Training"

    VAL_DIR = f"{dataset_directory}/Evaluation_Set/Evaluation_Set"
    VAL_LABELS = pd.read_csv(f"{VAL_DIR}/RFMiD_Validation_Labels.csv")
    VAL_LABELS = VAL_LABELS[["ID", "Disease_Risk"]]
    VAL_DATA = f"{VAL_DIR}/Validation"

    TEST_DIR = f"{dataset_directory}/Test_Set/Test_Set"
    TEST_LABELS = pd.read_csv(f"{TEST_DIR}/RFMiD_Testing_Labels.csv")
    TEST_LABELS = TEST_LABELS[["ID", "Disease_Risk"]]
    TEST_DATA = f"{TEST_DIR}/Test"

    dataset_train = CustomImageDataset(df=TRAIN_LABELS, img_dir=TRAIN_DATA, transform=TRAIN_TRANSFORMS)
    dataset_val = CustomImageDataset(df=VAL_LABELS, img_dir=VAL_DATA, transform=TEST_TRANSFORMS)
    dataset_test = CustomImageDataset(df=TEST_LABELS, img_dir=TEST_DATA, transform=TEST_TRANSFORMS)

    model = CNNGNNModel(opt=opt, cnn_blocks=opt.cnn_res_blocks_list, extraction_layer=opt.cnn_extraction_layer, gnn_channels=gnn_channels, k=opt.k_neighbours, drop_path=opt.stochastic_path).to(DEVICE)
    print(summary(model, input_size=(opt.batch_size, 3, 224, 224)))
    UseModel(opt, model, dataset_train, dataset_val, dataset_test, gnn_channels)
