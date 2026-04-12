"""
Retinal Fundus Image Classification Streamlit App
Supports binary and multilabel disease classification using CNN and GNN models.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import time

import utils
from utils import (
    load_test_labels,
    load_binary_model,
    load_multilabel_model,
    run_binary_inference,
    run_multilabel_inference,
    compare_binary_results,
    compare_multilabel_results,
    get_ground_truth,
    is_test_image,
    extract_image_id,
    prepare_image_for_model,
    DISEASE_NAMES,
    TEST_IMAGES_DIR,
)

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="Retinal Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stMetric { text-align: center; }
    .correct { color: green; }
    .incorrect { color: red; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==================== Session State Initialization ====================

if "test_labels" not in st.session_state:
    st.session_state.test_labels = load_test_labels()

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "inference_result" not in st.session_state:
    st.session_state.inference_result = None

if "image_id" not in st.session_state:
    st.session_state.image_id = None

if "is_test_image" not in st.session_state:
    st.session_state.is_test_image_flag = False

# ==================== Sidebar Configuration ====================

st.sidebar.title("🔬 Classification Settings")

classification_type = st.sidebar.radio(
    "Choose classification type:",
    ["Binary Classification", "Multilabel Classification"],
    horizontal=False,
)

model_type = st.sidebar.radio(
    "Choose model:", ["GNN", "CNN", "CNN-GNN"], horizontal=True
)

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown("""
    ### Retinal Classification System
    
    **Binary Classification**
    - Predicts overall disease risk (Yes/No)
    - Fast single prediction
    - Confidence score provided
    
    **Multilabel Classification**
    - Detects specific retinal diseases
    - Adjustable detection threshold
    - Shows up to 45 different conditions
    
    **Models**
    - **GNN**: Graph Neural Network approach
    - **CNN**: Convolutional Neural Network approach
    
    **Image Sources**
    - Test Set: Pre-loaded images with known labels
    - Custom Upload: Your own retinal images
    """)

st.sidebar.markdown("---")

# Display dataset info
with st.sidebar.expander("📊 Dataset Info", expanded=False):
    test_labels = st.session_state.test_labels
    if test_labels is not None:
        st.markdown(f"""
        **Test Set Statistics**
        - Total images: {len(test_labels)}
        - Available diseases: {len(utils.DISEASE_NAMES)}
        - Image size: 224×224
        """)
    else:
        st.warning("Could not load dataset information")

# ==================== Binary Classification Page ====================


def binary_classification():
    st.title("🔬 Binary Classification - Disease Risk")
    st.markdown("Predicts whether the patient has a disease risk or not.")

    # Get labels
    test_labels = st.session_state.test_labels

    # Image selection
    st.subheader("Select or Upload Image")

    tab1, tab2 = st.tabs(["Test Set", "Upload Custom"])

    with tab1:
        st.markdown("Browse the test set images below:")

        # Get list of test images
        test_image_files = sorted([f for f in TEST_IMAGES_DIR.glob("*.png")])[:12]

        if test_image_files:
            # Create 4-column grid
            cols_per_row = 4
            for i in range(0, len(test_image_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    if i + col_idx < len(test_image_files):
                        img_file = test_image_files[i + col_idx]
                        image_id = extract_image_id(img_file.name)

                        with col:
                            st.image(str(img_file), width=200)
                            st.caption(f"ID: {image_id}")
                            if st.button(f"Select", key=f"select_binary_{image_id}"):
                                st.session_state.selected_image = str(img_file)
                                st.session_state.image_id = image_id
                                st.session_state.is_test_image_flag = is_test_image(
                                    image_id, test_labels
                                )
                                st.rerun()
        else:
            st.warning("No test images found in dataset")

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus image", type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            st.session_state.selected_image = uploaded_file
            st.session_state.image_id = None
            st.session_state.is_test_image_flag = False

    # Display selected image
    if st.session_state.selected_image is not None:
        st.subheader("Selected Image")

        # Prepare image for display
        if isinstance(st.session_state.selected_image, str):
            display_image = prepare_image_for_model(
                st.session_state.selected_image, 224
            )
        else:
            from PIL import Image

            display_image = Image.open(st.session_state.selected_image).convert("RGB")
            display_image = display_image.resize((224, 224))

        col1, col2 = st.columns([1, 1])

        with col1:
            if display_image:
                st.image(display_image, caption="224x224 Input", width=300)

        with col2:
            image_info = (
                "Test Set Image"
                if st.session_state.is_test_image_flag
                else "User-Uploaded Image"
            )
            if st.session_state.image_id:
                image_info += f" (ID: {st.session_state.image_id})"

            st.markdown(f"**Image Source**: {image_info}")

        # Run Model Button
        if st.button(
            "🚀 Run Binary Classification Model",
            key="binary_run_btn",
            use_container_width=True,
        ):
            with st.spinner("Running inference..."):
                # Load model
                model = load_binary_model(model_type.lower())

                if model is None:
                    st.error(
                        f"Could not load {model_type} model. Weights may not be available yet."
                    )
                else:
                    # Run inference
                    if isinstance(st.session_state.selected_image, str):
                        image_path = st.session_state.selected_image
                    else:
                        # Save uploaded file temporarily
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".png"
                        ) as tmp:
                            tmp.write(st.session_state.selected_image.getbuffer())
                            image_path = tmp.name

                    result = run_binary_inference(model, image_path)

                    if result:
                        st.session_state.inference_result = result
                        st.success("Inference complete!")
                    else:
                        st.error("Inference failed")

        # Display Results
        if st.session_state.inference_result is not None:
            result = st.session_state.inference_result

            st.markdown("---")
            st.subheader("📊 Results")

            # Get ground truth if available
            ground_truth = None
            if st.session_state.is_test_image_flag:
                ground_truth = get_ground_truth(st.session_state.image_id, test_labels)

            comparison = compare_binary_results(
                result["disease_risk"], result["probability"], ground_truth
            )

            # Display prediction
            col1, col2, col3 = st.columns(3)

            with col1:
                pred_text = "YES ✓" if comparison["prediction"] == 1 else "NO ✗"
                st.metric("Disease Risk", pred_text)

            with col2:
                st.metric("Confidence", f"{comparison['probability']:.4f}")

            with col3:
                st.metric("Inference Time", f"{result['inference_time']:.3f}s")

            # Display ground truth comparison
            if not comparison["is_user_upload"]:
                st.markdown("---")
                st.subheader("🎯 Ground Truth Comparison")

                gt_text = "YES" if comparison["ground_truth"] == 1 else "NO"
                match_text = "✓ CORRECT" if comparison["match"] else "✗ INCORRECT"
                match_color = "green" if comparison["match"] else "red"

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Actual Label**: {gt_text}")

                with col2:
                    st.markdown(
                        f"**Match**: <span style='color:{match_color}'>{match_text}</span>",
                        unsafe_allow_html=True,
                    )


# ==================== Multilabel Classification Page ====================


def multilabel_classification():
    st.title("🔬 Multilabel Classification - Disease Detection")
    st.markdown("Detects specific retinal diseases from the image.")

    # Get labels
    test_labels = st.session_state.test_labels

    # Threshold slider
    st.subheader("Detection Settings")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only show diseases with probability above this threshold",
    )

    # Image selection
    st.subheader("Select or Upload Image")

    tab1, tab2 = st.tabs(["Test Set", "Upload Custom"])

    with tab1:
        st.markdown("Browse the test set images below:")

        # Get list of test images
        test_image_files = sorted([f for f in TEST_IMAGES_DIR.glob("*.png")])[:12]

        if test_image_files:
            # Create 4-column grid
            cols_per_row = 4
            for i in range(0, len(test_image_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    if i + col_idx < len(test_image_files):
                        img_file = test_image_files[i + col_idx]
                        image_id = extract_image_id(img_file.name)

                        with col:
                            st.image(str(img_file), width=200)
                            st.caption(f"ID: {image_id}")
                            if st.button(
                                f"Select", key=f"select_multilabel_{image_id}"
                            ):
                                st.session_state.selected_image = str(img_file)
                                st.session_state.image_id = image_id
                                st.session_state.is_test_image_flag = is_test_image(
                                    image_id, test_labels
                                )
                                st.rerun()
        else:
            st.warning("No test images found in dataset")

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus image",
            type=["png", "jpg", "jpeg"],
            key="multilabel_upload",
        )

        if uploaded_file is not None:
            st.session_state.selected_image = uploaded_file
            st.session_state.image_id = None
            st.session_state.is_test_image_flag = False

    # Display selected image
    if st.session_state.selected_image is not None:
        st.subheader("Selected Image")

        # Prepare image for display
        if isinstance(st.session_state.selected_image, str):
            display_image = prepare_image_for_model(
                st.session_state.selected_image, 224
            )
        else:
            from PIL import Image

            display_image = Image.open(st.session_state.selected_image).convert("RGB")
            display_image = display_image.resize((224, 224))

        col1, col2 = st.columns([1, 1])

        with col1:
            if display_image:
                st.image(display_image, caption="224x224 Input", width=300)

        with col2:
            image_info = (
                "Test Set Image"
                if st.session_state.is_test_image_flag
                else "User-Uploaded Image"
            )
            if st.session_state.image_id:
                image_info += f" (ID: {st.session_state.image_id})"

            st.markdown(f"**Image Source**: {image_info}")
            st.markdown(f"**Detection Threshold**: {threshold:.2f}")

        # Run Model Button
        if st.button(
            "🚀 Run Multilabel Classification Model",
            key="multilabel_run_btn",
            use_container_width=True,
        ):
            with st.spinner("Running inference..."):
                # Load model
                model = load_multilabel_model(model_type.lower())

                if model is None:
                    st.error(
                        f"Could not load {model_type} model. Weights may not be available yet."
                    )
                else:
                    # Run inference
                    if isinstance(st.session_state.selected_image, str):
                        image_path = st.session_state.selected_image
                    else:
                        # Save uploaded file temporarily
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".png"
                        ) as tmp:
                            tmp.write(st.session_state.selected_image.getbuffer())
                            image_path = tmp.name

                    result = run_multilabel_inference(model, image_path, threshold)

                    if result:
                        st.session_state.inference_result = result
                        st.success("Inference complete!")
                    else:
                        st.error("Inference failed")

        # Display Results
        if st.session_state.inference_result is not None:
            result = st.session_state.inference_result

            st.markdown("---")
            st.subheader("📊 Results")

            # Metadata
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Diseases Detected", len(result["detected"]))

            with col2:
                st.metric("Diseases Not Detected", len(result["undetected"]))

            with col3:
                st.metric("Inference Time", f"{result['inference_time']:.3f}s")

            # Detected diseases table
            if result["detected"]:
                st.markdown("---")
                st.subheader("🔴 Detected Diseases")

                # Get ground truth if available
                ground_truth = None
                if st.session_state.is_test_image_flag:
                    ground_truth = get_ground_truth(
                        st.session_state.image_id, test_labels
                    )

                # Build table data
                table_data = []
                for detection in result["detected"]:
                    disease = detection["disease"]
                    prob = detection["probability"]

                    if ground_truth:
                        actual = (
                            "YES"
                            if ground_truth["diseases"].get(disease, 0) == 1
                            else "NO"
                        )
                        match = (
                            "✓"
                            if (
                                prob > 0.5
                                and ground_truth["diseases"].get(disease, 0) == 1
                            )
                            else "✗"
                        )
                        table_data.append(
                            {
                                "Disease": disease,
                                "Probability": f"{prob:.4f}",
                                "Actual": actual,
                                "Match": match,
                            }
                        )
                    else:
                        table_data.append(
                            {
                                "Disease": disease,
                                "Probability": f"{prob:.4f}",
                                "Actual": "—",
                                "Match": "—",
                            }
                        )

                df_detected = pd.DataFrame(table_data)
                st.dataframe(df_detected, use_container_width=True, hide_index=True)
            else:
                st.info("No diseases detected above the threshold")

            # Not detected summary
            if result["undetected"]:
                st.markdown("---")
                st.subheader("🟢 Not Detected (Below Threshold)")

                undetected_text = ", ".join(result["undetected"])
                st.markdown(
                    f"{len(result['undetected'])} diseases below threshold:\n\n{undetected_text}"
                )


# ==================== Main App ====================


def main():
    # Title and header
    st.markdown("# 🔬 Retinal Fundus Image Classification")
    st.markdown("""
    ### CNN vs GNN Models for Disease Detection
    
    This application uses advanced deep learning models to analyse retinal fundus images 
    for disease classification. Select a classification type and model from the sidebar to begin.
    """)
    st.markdown("---")

    # Show model status info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Classification Type", value=classification_type.split()[0])

    with col2:
        st.metric(label="Model", value=model_type)

    with col3:
        test_labels = st.session_state.test_labels
        status = "✓ Ready" if test_labels is not None else "⚠ Check Data"
        st.metric(label="Data Status", value=status)

    st.markdown("---")

    # Route to appropriate page
    if classification_type == "Binary Classification":
        binary_classification()
    else:
        multilabel_classification()


if __name__ == "__main__":
    main()
