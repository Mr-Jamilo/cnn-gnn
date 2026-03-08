import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from datetime import datetime

TRAIN_DIR = 'dataset/Training_Set/Training_Set'
TRAIN_LABELS = pd.read_csv(f'{TRAIN_DIR}/RFMiD_Training_Labels.csv')
TRAIN_LABELS = TRAIN_LABELS[['ID', 'DR', 'MH', 'TSLN', 'ODC']]
TRAIN_DATA = f'{TRAIN_DIR}/Training'

VAL_DIR = 'dataset/Evaluation_Set/Evaluation_Set'
VAL_LABELS = pd.read_csv(f'{VAL_DIR}/RFMiD_Validation_Labels.csv')
VAL_LABELS = VAL_LABELS[['ID', 'DR', 'MH', 'TSLN', 'ODC']]
VAL_DATA = f'{VAL_DIR}/Validation'

TEST_DIR = 'dataset/Test_Set/Test_Set'
TEST_LABELS = pd.read_csv(f'{TEST_DIR}/RFMiD_Testing_Labels.csv')
TEST_LABELS = TEST_LABELS[['ID', 'DR', 'MH', 'TSLN', 'ODC']]
TEST_DATA = f'{TEST_DIR}/Test'
RES_BLOCKS = [2, 2, 2, 2]
NUM_CLASSES = 4
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 4e-5
USE_WEIGHT_BIAS = False
EPOCHS = 150
THRESHOLD = 0.8
TRAINING_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU available: {physical_devices}")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU available, using CPU")

def create_data_generator(df, img_dir, batch_size, training=False):
    """Create a TensorFlow data generator for the dataset"""
    def data_generator():
        indices = np.arange(len(df))
        if training:
            np.random.shuffle(indices)

        for idx in indices:
            row = df.iloc[idx]
            img_name = str(row['ID']) + ".png"
            img_path = os.path.join(img_dir, img_name)

            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0  # Normalize to [0, 1]

            # Normalize with ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std

            labels = row.drop('ID').values.astype(np.float32)

            yield img, labels

    output_signature = (
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )

    if training:
        # Apply data augmentation
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            return image, label

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def residual_block(x, filters, stride=1, name=None):
    """Residual block for ResNet"""
    shortcut = x

    # First convolution
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    # Second convolution
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, name=f'{name}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu2')(x)

    return x

def create_resnet(num_blocks, num_classes=NUM_CLASSES):
    """Create ResNet model using Keras functional API"""
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Initial convolution
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)

    # ResNet layers
    filters_list = [64, 128, 256, 512]
    strides_list = [1, 2, 2, 2]

    for layer_idx, (num_blocks_in_layer, filters, stride) in enumerate(zip(num_blocks, filters_list, strides_list)):
        for block_idx in range(num_blocks_in_layer):
            block_stride = stride if block_idx == 0 else 1
            x = residual_block(x, filters, stride=block_stride, name=f'layer{layer_idx+1}_block{block_idx}')

    # Global average pooling and output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation=None, name='output')(x)  # No activation, we'll use BCEWithLogitsLoss equivalent

    model = Model(inputs=inputs, outputs=outputs, name='ResNet')
    return model

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.get_weights()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = model.get_weights()
            self.counter = 0

    def load_best_model(self, model):
        if self.best_weights is not None:
            model.set_weights(self.best_weights)

def PrepData(train_df, val_df, test_df):
    """Prepare data loaders and compute class weights"""
    label_cols = [col for col in train_df.columns if col != 'ID']
    positives = train_df[label_cols].sum(axis=0).astype(float)
    total = len(train_df)
    negatives = total - positives
    pos_weight_vals = (negatives / positives).values
    pos_weights = tf.constant(pos_weight_vals, dtype=tf.float32)
    print("Pos weights:", pos_weights.numpy())

    train_dataset = create_data_generator(train_df, TRAIN_DATA, TRAINING_BATCH_SIZE, training=True)
    val_dataset = create_data_generator(val_df, VAL_DATA, TEST_BATCH_SIZE, training=False)
    test_dataset = create_data_generator(test_df, TEST_DATA, TEST_BATCH_SIZE, training=False)

    return train_dataset, val_dataset, test_dataset, pos_weights



def train_one_epoch(dataset, model, loss_fn, optimizer):
    """Train for one epoch"""
    loss_list = []
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    batch_num = 0
    for images, labels in dataset:
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)

        # Update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_list.append(loss.numpy())

        # Calculate predictions
        probs = tf.nn.sigmoid(logits)
        preds = tf.cast(probs > THRESHOLD, tf.float32)

        correct += tf.reduce_sum(tf.cast(preds == labels, tf.float32)).numpy()
        total += tf.size(labels, out_type=tf.float32).numpy()

        all_preds.append(preds.numpy())
        all_targets.append(labels.numpy())

        batch_num += 1
        print(f"  Batch {batch_num} - Loss: {loss.numpy():.4f} - Acc: {(correct/total):.4f}", end='\r')

    avg_loss = np.mean(loss_list)
    avg_accuracy = correct / total

    # Calculate F1, precision, recall
    all_preds_np = np.vstack(all_preds)
    all_targets_np = np.vstack(all_targets)
    f1 = f1_score(all_targets_np, all_preds_np, average='samples', zero_division=0)
    precision = precision_score(all_targets_np, all_preds_np, average='samples', zero_division=0)
    recall = recall_score(all_targets_np, all_preds_np, average='samples', zero_division=0)

    return avg_loss, avg_accuracy, f1, precision, recall

def TestModel(model, loss_fn, dataset):
    """Test the model and return metrics"""
    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    for images, labels in dataset:
        logits = model(images, training=False)
        loss = loss_fn(labels, logits)
        test_loss += loss.numpy()

        probs = tf.nn.sigmoid(logits)
        preds = tf.cast(probs > THRESHOLD, tf.float32)

        correct += tf.reduce_sum(tf.cast(preds == labels, tf.float32)).numpy()
        total += tf.size(labels, out_type=tf.float32).numpy()

        all_preds.append(preds.numpy())
        all_targets.append(labels.numpy())

    # Calculate metrics
    batch_count = len(all_preds)
    avg_loss = test_loss / batch_count if batch_count > 0 else 0
    accuracy = correct / total if total > 0 else 0

    all_preds_np = np.vstack(all_preds)
    all_targets_np = np.vstack(all_targets)

    f1 = f1_score(all_targets_np, all_preds_np, average='samples', zero_division=0)
    precision = precision_score(all_targets_np, all_preds_np, average='samples', zero_division=0)
    recall = recall_score(all_targets_np, all_preds_np, average='samples', zero_division=0)

    print(classification_report(all_targets_np, all_preds_np, zero_division=0))

    return avg_loss, accuracy, f1, precision, recall


def UseModel(model, dataset_train, dataset_val, dataset_test):
    """Train and evaluate the model"""
    # Create custom loss function with pos_weights
    train_dataloader, val_dataloader, test_dataloader, pos_weights = PrepData(dataset_train, dataset_val, dataset_test)

    def weighted_bce_loss(y_true, y_pred):
        """Binary cross entropy loss with pos_weights"""
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        if USE_WEIGHT_BIAS:
            weighted_bce = bce * pos_weights
        else:
            weighted_bce = bce
        return tf.reduce_mean(weighted_bce)

    # Create optimizer (AdamW for weight decay support)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Early stopping
    early_stopping = EarlyStopping(patience=25, delta=0)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []

    actual_epochs = 0
    best_val_loss = np.inf

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(
            train_dataloader, model, weighted_bce_loss, optimizer
        )

        # Validate
        val_loss, val_acc, val_f1, val_precision, val_recall = TestModel(
            model, weighted_bce_loss, val_dataloader
        )

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)

        # Check early stopping
        early_stopping(val_loss, model)
        actual_epochs += 1

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val   F1: {val_f1:.4f}, Val   Precision: {val_precision:.4f}, Val   Recall: {val_recall:.4f}")

    print("\nTraining complete!")

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_graph.png")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy_graph.png")

    # Plot F1
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.title('F1 over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig("f1_graph.png")

    # Load best model and test
    early_stopping.load_best_model(model)
    test_loss, test_acc, test_f1, test_precision, test_recall = TestModel(model, weighted_bce_loss, test_dataloader)

    print(f'\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')

    # Logging
    summary_path = 'main.txt'
    header = "date;time;res_blocks;classes;learning_rate;weight_decay;weight_parameter;Threshold;epochs;early_stopping;train_transforms;test_transforms;precision;recall;f1_score\n"

    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    label_cols = [c for c in dataset_train.columns if c != 'ID']
    classes_str = ",".join(label_cols)

    weight_param_used = USE_WEIGHT_BIAS
    early_stopping_used = early_stopping.early_stop

    train_transforms_str = "RandomHorizontalFlip,RandomRotation"
    test_transforms_str = "None"

    line = (
        f"{date_str};"
        f"{time_str};"
        f"{RES_BLOCKS};"
        f"{len(label_cols)}({classes_str});"
        f"{LEARNING_RATE};"
        f"{WEIGHT_DECAY};"
        f"{str(weight_param_used)};"
        f"{THRESHOLD};"
        f"{actual_epochs};"
        f"{str(early_stopping_used)};"
        f"{train_transforms_str};"
        f"{test_transforms_str};"
        f"{test_precision:.4f};"
        f"{test_recall:.4f};"
        f"{test_f1:.4f}\n"
    )

    write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(line)

    print(f"\nResults logged to {summary_path}")

if __name__ == '__main__':
    # Check GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("Warning: No GPU available, using CPU")

    # Create model
    model = create_resnet(RES_BLOCKS, num_classes=NUM_CLASSES)

    # Print model summary
    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3))
    model.summary()

    # Train model
    UseModel(model, TRAIN_LABELS, VAL_LABELS, TEST_LABELS)
