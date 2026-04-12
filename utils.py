"""
Utility functions for the Streamlit retinal classification app.
Handles model loading, CSV data loading, image processing, and inference.
"""

import os
import argparse
import time
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
import streamlit as st

# All disease column names from the CSV
DISEASE_NAMES = [
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "TV",
    "AH",
    "ODP",
    "ODE",
    "ST",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
    "RP",
    "CWS",
    "CB",
    "ODPM",
    "PRH",
    "MNF",
    "HR",
    "CRAO",
    "TD",
    "CME",
    "PTCR",
    "CF",
    "VH",
    "MCA",
    "VS",
    "BRAO",
    "PLQ",
    "HPED",
    "CL",
]

# Paths
WEIGHTS_DIR = Path("weights")
DATASET_DIR = Path("dataset")
TEST_DIR = DATASET_DIR / "Test_Set" / "Test_Set"
TEST_IMAGES_DIR = TEST_DIR / "Test"
TEST_LABELS_PATH = TEST_DIR / "RFMiD_Testing_Labels.csv"

VAL_DIR = DATASET_DIR / "Evaluation_Set" / "Evaluation_Set"
VAL_LABELS_PATH = VAL_DIR / "RFMiD_Validation_Labels.csv"

TRAIN_DIR = DATASET_DIR / "Training_Set" / "Training_Set"
TRAIN_LABELS_PATH = TRAIN_DIR / "RFMiD_Training_Labels.csv"

# Model weight paths
WEIGHTS_PATHS = {
    "binary": {
        "gnn": WEIGHTS_DIR / "binary" / "vignn.pth",
        "cnn": WEIGHTS_DIR / "binary" / "cnn.pth",
        "cnn-gnn": WEIGHTS_DIR / "binary" / "cnn-gnn.pth",
    },
    "multilabel": {
        "gnn": WEIGHTS_DIR / "multilabel" / "vignn.pth",
        "cnn": WEIGHTS_DIR / "multilabel" / "cnn.pth",
        "cnn-gnn": WEIGHTS_DIR / "multilabel" / "cnn-gnn.pth",
    },
}


# ==================== CSV/Data Loading ====================


@st.cache_resource
def load_test_labels() -> Optional[pd.DataFrame]:
    """Load test set labels from CSV."""
    try:
        if not TEST_LABELS_PATH.exists():
            st.warning(f"Test labels file not found: {TEST_LABELS_PATH}")
            return None
        df = pd.read_csv(TEST_LABELS_PATH)
        df = df.set_index("ID")
        return df
    except Exception as e:
        st.error(f"Error loading test labels: {e}")
        return None


@st.cache_resource
def load_validation_labels() -> Optional[pd.DataFrame]:
    """Load validation set labels from CSV."""
    try:
        if not VAL_LABELS_PATH.exists():
            st.warning(f"Validation labels file not found: {VAL_LABELS_PATH}")
            return None
        df = pd.read_csv(VAL_LABELS_PATH)
        df = df.set_index("ID")
        return df
    except Exception as e:
        st.error(f"Error loading validation labels: {e}")
        return None


@st.cache_resource
def load_training_labels() -> Optional[pd.DataFrame]:
    """Load training set labels from CSV."""
    try:
        if not TRAIN_LABELS_PATH.exists():
            st.warning(f"Training labels file not found: {TRAIN_LABELS_PATH}")
            return None
        df = pd.read_csv(TRAIN_LABELS_PATH)
        df = df.set_index("ID")
        return df
    except Exception as e:
        st.error(f"Error loading training labels: {e}")
        return None


def get_ground_truth(
    image_id: int, labels_df: Optional[pd.DataFrame]
) -> Optional[Dict]:
    """
    Get ground truth labels for an image ID.

    Returns:
        Dict with 'disease_risk' (0/1) and 'diseases' (dict of disease labels)
        or None if not found
    """
    if labels_df is None or image_id not in labels_df.index:
        return None

    try:
        row = labels_df.loc[image_id]
        ground_truth = {
            "disease_risk": int(row["Disease_Risk"]),
            "diseases": {
                disease: int(row[disease])
                for disease in DISEASE_NAMES
                if disease in row.index
            },
        }
        return ground_truth
    except Exception as e:
        st.warning(f"Error getting ground truth for ID {image_id}: {e}")
        return None


def is_test_image(image_id: int, labels_df: Optional[pd.DataFrame]) -> bool:
    """Check if image ID exists in test set labels."""
    if labels_df is None:
        return False
    return image_id in labels_df.index


# ==================== Image Processing ====================


def extract_image_id(filename: str) -> Optional[int]:
    """Extract numeric ID from filename (e.g., '1.png' -> 1)."""
    try:
        name = Path(filename).stem
        return int(name)
    except (ValueError, AttributeError):
        return None


def validate_image_file(uploaded_file) -> bool:
    """Check if uploaded file is a valid image format."""
    valid_formats = {"image/png", "image/jpeg"}
    return uploaded_file.type in valid_formats


def prepare_image_for_model(
    image_path, target_size: int = 224
) -> Optional[Image.Image]:
    """
    Load and resize image to target size.

    Returns:
        PIL Image resized to (target_size, target_size) or None if error
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# ==================== Model Loading ====================


def load_binary_model(model_type: str = "gnn"):
    """
    Load a binary classification model.

    Args:
        model_type: "gnn", "cnn", or "cnn-gnn"

    Returns:
        Loaded model or None if weights not found
    """
    try:
        if model_type not in ["gnn", "cnn", "cnn-gnn"]:
            st.error(
                f"Unknown model type: {model_type}. Must be 'gnn', 'cnn', or 'cnn-gnn'"
            )
            return None

        weights_path = WEIGHTS_PATHS["binary"][model_type]

        if not weights_path.exists():
            st.warning(f"Binary {model_type.upper()} weights not found: {weights_path}")
            return None

        # Import here to avoid CUDA assertions during app startup
        from models.binary.gnn import ViGNN, DEVICE

        # Create model
        opt = argparse.Namespace(
            graph_layer_type="GCN", k_neighbours=5, stochastic_path=0.1
        )

        if model_type == "gnn":
            model = ViGNN(
                opt,
                in_channels=3,
                num_classes=1,
                k=opt.k_neighbours,
                depths=[2, 2, 6, 2],
                channels=[80, 160, 400, 640],
                drop_path=opt.stochastic_path,
            ).to(DEVICE)

            # Load weights
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            return model

        elif model_type in ["cnn", "cnn-gnn"]:
            st.warning(f"{model_type.upper()} model loading not yet implemented")
            return None

    except Exception as e:
        st.error(f"Error loading binary {model_type.upper()} model: {e}")
        return None
        opt = argparse.Namespace(
            graph_layer_type="GCN", k_neighbours=5, stochastic_path=0.1
        )

        if model_type == "gnn":
            model = ViGNN(
                opt,
                in_channels=3,
                num_classes=1,
                k=opt.k_neighbours,
                depths=[2, 2, 6, 2],
                channels=[80, 160, 400, 640],
                drop_path=opt.stochastic_path,
            ).to(DEVICE)

            # Load weights
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            return model

        elif model_type == "cnn":
            st.warning("CNN model loading not yet implemented")
            return None

    except Exception as e:
        st.error(f"Error loading binary {model_type.upper()} model: {e}")
        return None


def load_multilabel_model(model_type: str = "gnn"):
    """
    Load a multilabel classification model.

    Args:
        model_type: "gnn", "cnn", or "cnn-gnn"

    Returns:
        Loaded model or None if weights not found
    """
    try:
        if model_type not in ["gnn", "cnn", "cnn-gnn"]:
            st.error(f"Unknown model type: {model_type}. Must be 'gnn', 'cnn', or 'cnn-gnn'")
            return None

        weights_path = WEIGHTS_PATHS["multilabel"][model_type]

        if not weights_path.exists():
            st.warning(
                f"Multilabel {model_type.upper()} weights not found: {weights_path}"
            )
            return None

        # Import here to avoid CUDA assertions during app startup
        from models.multilabel.gnn import ViGNN, DEVICE

        # Create model
        opt = argparse.Namespace(
            graph_layer_type="GCN", k_neighbours=5, stochastic_path=0.1
        )

        if model_type == "gnn":
            model = ViGNN(
                opt,
                in_channels=3,
                num_classes=len(DISEASE_NAMES),  # 45 diseases
                k=opt.k_neighbours,
                depths=[2, 2, 6, 2],
                channels=[80, 160, 400, 640],
                drop_path=opt.stochastic_path,
            ).to(DEVICE)

            # Load weights
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            return model

        elif model_type in ["cnn", "cnn-gnn"]:
            st.warning(f"{model_type.upper()} model loading not yet implemented")
            return None

    except Exception as e:
        st.error(f"Error loading multilabel {model_type.upper()} model: {e}")


# ==================== Inference ====================


def run_binary_inference(model, image_path) -> Optional[Dict]:
    """
    Run binary classification inference.

    Returns:
        Dict with prediction, probability, and inference time
        or None if error
    """
    try:
        import torch
        from models.binary.gnn import TEST_TRANSFORMS, DEVICE
        from PIL import Image

        model.eval()
        start_time = time.time()

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = TEST_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0

        inference_time = time.time() - start_time

        return {
            "disease_risk": prediction,
            "probability": round(probability, 4),
            "inference_time": inference_time,
        }

    except Exception as e:
        st.error(f"Error running inference: {e}")
        return None


def run_multilabel_inference(
    model, image_path, threshold: float = 0.5
) -> Optional[Dict]:
    """
    Run multilabel classification inference.

    Returns:
        Dict with detected diseases, probabilities, and inference time
        or None if error
    """
    try:
        from models.multilabel.gnn import TEST_TRANSFORMS, DEVICE
        from PIL import Image

        model.eval()

        start_time = time.time()

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = TEST_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.sigmoid(output).squeeze(0).cpu().numpy()

        inference_time = time.time() - start_time

        # Create results with only detected diseases (above threshold)
        detected = []
        for disease_name, prob in zip(DISEASE_NAMES, probabilities):
            if prob >= threshold:
                detected.append(
                    {"disease": disease_name, "probability": round(float(prob), 4)}
                )

        # Sort by probability descending
        detected.sort(key=lambda x: x["probability"], reverse=True)

        # Get undetected diseases
        detected_names = {d["disease"] for d in detected}
        undetected = [name for name in DISEASE_NAMES if name not in detected_names]

        return {
            "detected": detected,
            "undetected": undetected,
            "all_probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(DISEASE_NAMES, probabilities)
            },
            "inference_time": inference_time,
            "threshold": threshold,
        }

    except Exception as e:
        st.error(f"Error running multilabel inference: {e}")
        return None


# ==================== Results Comparison ====================


def compare_binary_results(
    prediction: int, probability: float, ground_truth: Optional[Dict]
) -> Dict:
    """
    Compare binary prediction with ground truth.

    Returns:
        Dict with prediction, ground_truth, and match status
    """
    if ground_truth is None:
        return {
            "prediction": prediction,
            "probability": probability,
            "ground_truth": None,
            "match": None,
            "is_user_upload": True,
        }

    gt_risk = ground_truth["disease_risk"]
    match = prediction == gt_risk

    return {
        "prediction": prediction,
        "probability": probability,
        "ground_truth": gt_risk,
        "match": match,
        "is_user_upload": False,
    }


def compare_multilabel_results(
    detected: List[Dict], ground_truth: Optional[Dict], threshold: float
) -> Dict:
    """
    Compare multilabel predictions with ground truth.

    Returns:
        Dict with detected diseases, ground truth, and match status
    """
    if ground_truth is None:
        return {
            "detected": detected,
            "ground_truth": None,
            "comparison": None,
            "is_user_upload": True,
            "threshold": threshold,
        }

    # Build comparison table
    comparison = []
    gt_diseases = ground_truth["diseases"]

    detected_names = {d["disease"] for d in detected}

    for disease_name in DISEASE_NAMES:
        prob = next(
            (d["probability"] for d in detected if d["disease"] == disease_name), None
        )
        gt_label = gt_diseases.get(disease_name, 0)

        if disease_name in detected_names:
            # Detected by model
            match = (prob is not None and prob > 0.5) and gt_label == 1
            comparison.append(
                {
                    "disease": disease_name,
                    "probability": prob,
                    "ground_truth": gt_label,
                    "match": match,
                    "detected": True,
                }
            )

    return {
        "detected": detected,
        "ground_truth": ground_truth["diseases"],
        "comparison": comparison,
        "is_user_upload": False,
        "threshold": threshold,
    }
