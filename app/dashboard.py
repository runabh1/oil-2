# app/dashboard.py

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Must import the model structure and prediction logic
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_definition import build_unet
from src.prediction_service import preprocess_and_predict, colorize_mask 

# Configuration
st.set_page_config(layout="wide", page_title="InterpAI Portfolio")

# Define paths and parameters
MODEL_PATH = 'models/trained_interpai_unet_model.h5'
TEST_DATA_DIR = 'data/test_samples/' 
IMG_SIZE = 128
CLASSES = 10

# --- 1. Load Model (Cached) ---
# Use the custom build_unet function to load the weights correctly
@st.cache_resource
def load_the_model():
    """Load the trained Keras model and weights."""
    try:
        # Build structure
        model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=CLASSES)
        
        # Load weights
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}. Check your 'models/' folder.")
            return None
            
        # load_weights is necessary since we built the model structure manually
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model architecture or weights: {e}")
        return None

model = load_the_model()

if model:
    st.title("🌊 AI-Driven Seismic Interpretation (InterpAI)")
    st.markdown("### Automated Fault and Horizon Segmentation for E&P")

    # --- Sidebar File Selector ---
    try:
        test_files = sorted([f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.tif') or f.endswith('.tiff')])
        if not test_files:
             st.error(f"Error: No TIFF files found in '{TEST_DATA_DIR}'.")
             st.stop()
        
        selected_file = st.sidebar.selectbox("Select Test Seismic Slice:", test_files)
    except FileNotFoundError:
         st.error(f"Error: The directory '{TEST_DATA_DIR}' was not found. Check Step 1 setup.")
         st.stop()
    
    # Determine file paths
    tiff_path = os.path.join(TEST_DATA_DIR, selected_file)
    base_name = os.path.splitext(selected_file.split('_')[-1])[0] # e.g., 'Inline_050.tif' -> '050'
    mask_path = os.path.join(TEST_DATA_DIR, base_name + '.png') # Assumes true mask is '050.png'

    # Run Prediction
    img_processed, mask_pred, mask_true = preprocess_and_predict(model, tiff_path, mask_path)

    if img_processed is not None:
        
        col1, col2, col3 = st.columns(3)

        # 1. Original Image (Input)
        with col1:
            st.subheader("1. Original Seismic Input")
            # Display the processed, normalized grayscale input
            st.image(img_processed, caption=selected_file, use_column_width=True, clamp=True)
        
        # 2. True Mask (Ground Truth)
        with col2:
            st.subheader("2. Geologist's True Mask")
            if mask_true is not None:
                colored_true_mask = colorize_mask(mask_true)
                st.image(colored_true_mask, caption="Ground Truth (PNG Mask)", use_column_width=True)
            else:
                st.warning("True Mask (.png) not found.")

        # 3. Predicted Mask (AI Output)
        with col3:
            st.subheader("3. AI Predicted Mask")
            colored_pred_mask = colorize_mask(mask_pred)
            st.image(colored_pred_mask, caption="U-Net Segmentation Output", use_column_width=True)
        
        # --- Final Metrics and Impact Statement ---
        st.markdown("---")
        
        # Count the number of unique features identified by the AI
        features_found = np.unique(mask_pred).size
        st.info(f"Features Identified by AI: **{features_found}** out of {CLASSES} expected classes.")
        st.success("✅ **Portfolio Impact:** Automated interpretation drastically reduces manual analysis time from hours to minutes, enabling faster exploration decisions and improved reservoir characterization.")
        
        # Additional metrics
        if mask_true is not None:
            # Calculate accuracy metrics
            accuracy = np.mean(mask_pred == mask_true)
            st.metric("Segmentation Accuracy", f"{accuracy:.2%}")
            
            # Feature comparison
            true_features = np.unique(mask_true).size
            st.metric("True Features Detected", true_features)
            st.metric("AI Features Predicted", features_found)
        
        # Technical details
        with st.expander("Technical Details"):
            st.write(f"**Model Architecture:** U-Net with {IMG_SIZE}x{IMG_SIZE} input resolution")
            st.write(f"**Classes:** {CLASSES} geological features")
            st.write(f"**Input File:** {selected_file}")
            st.write(f"**Processing:** Normalized grayscale seismic data")
            
    else:
        st.error("Failed to process the selected file. Please check the file format and try again.")

else:
    st.error("Model could not be loaded. Please ensure the trained model file is in the models/ directory.")
