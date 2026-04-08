import streamlit as st
import pandas as pd
import torch
import argparse
import models.binary.gnn as gnn
from PIL import Image
from models.binary.gnn import ViGNN, DEVICE
from streamlit_image_select import image_select

st.write("#### 👈 Change classification type from the dropdown on the left.")

grid_images = [f"./dataset/Test_Set/Test_Set/Test/{i}.png" for i in range(1, 13)]
selected_img = image_select("Select a image to classify", grid_images)
img = st.write(selected_img)

uploaded_file = st.file_uploader("Upload A Retinal Fundus Image", type=["jpg", "jpeg", "png"])

opt = argparse.Namespace(graph_layer_type='GCN', k_neighbours=5, stochastic_path=0.1)
vignn_binary = ViGNN(opt, in_channels=3, num_classes=1, k=opt.k_neighbours, depths=[2, 2, 6, 2], channels=[80, 160, 400, 640], drop_path=opt.stochastic_path).to(DEVICE)
#vignn_binary.load_state_dict(torch.load("weights/binary/vignn.pth"))

el(image):
    st.write(gnn.predict_image(vignn_binary, image))

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).resize((224, 224))
    st.toast("SUCCESSFUL UPLOAD", icon="✅")
    st.image(uploaded_image, caption="224x224")

def binary():
    import streamlit as st

    st.write("# Binary Classification")
    model = st.selectbox("Select a Model", ("ViGNN", "CNN"))

    if selected_img is not None and uploaded_file is not None:
        st.write('#### Which image would you like to submit to the model?')
    else:
        st.image(selected_img, caption='retinal image')
        st.button('Run Model', on_click=run_model(selected_img))

def multiclass():
    import streamlit as st

    st.write("# Multiclass Classification")
    model = st.selectbox("Select a Model", ("ViGNN", "CNN"))

page_names_to_funcs = {"Binary": binary, "Multiclass": multiclass}
classification_type = st.sidebar.selectbox("Choose a classification", page_names_to_funcs.keys())
page_names_to_funcs[classification_type]()
