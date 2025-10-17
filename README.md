# 🌊 AI-Driven Seismic Interpretation (InterpAI)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io)


**Automated Fault and Horizon Segmentation for Exploration & Production (E&P)**

A sophisticated machine learning application that uses deep learning to automatically identify geological features in seismic data, revolutionizing traditional seismic interpretation workflows in the oil and gas industry.

## 🎯 Overview

InterpAI leverages a trained U-Net convolutional neural network to perform pixel-level segmentation of seismic images, automatically detecting and classifying geological features such as faults, horizons, salt domes, and reservoir zones. This application demonstrates how artificial intelligence can accelerate exploration workflows and improve interpretation consistency.

## ✨ Key Features

- **🤖 Automated Feature Detection**: AI-powered identification of 10 different geological feature classes
- **⚡ Real-time Processing**: Sub-second prediction times for seismic image analysis
- **📊 Interactive Dashboard**: Three-column comparison interface showing input, ground truth, and AI predictions
- **🎨 Color-coded Visualization**: Intuitive color mapping for different geological features
- **📈 Performance Metrics**: Real-time accuracy calculations and feature detection statistics
- **🔄 File Management**: Easy selection and processing of multiple seismic slices

## 🏗️ Architecture

### Deep Learning Model
- **Architecture**: U-Net Convolutional Neural Network
- **Input Resolution**: 128×128 pixels (grayscale seismic images)
- **Output Classes**: 10 geological feature types
- **Framework**: TensorFlow/Keras with optimized CPU inference

### Application Stack
- **Frontend**: Streamlit web application
- **Backend**: Python with TensorFlow for AI inference
- **Data Processing**: OpenCV, scikit-image, rasterio for geospatial data
- **Visualization**: Matplotlib for color mapping and image display

## 📁 Project Structure

```
oil-2/
├── app/
│   ├── dashboard.py          # Main Streamlit application
│   └── requirements.txt      # Python dependencies
├── src/
│   ├── model_definition.py  # U-Net architecture implementation
│   ├── prediction_service.py # Data processing and inference pipeline
│   └── __init__.py          # Package initialization
├── models/
│   └── trained_interpai_unet_model.h5  # Pre-trained model weights
├── data/
│   └── test_samples/        # Sample seismic data and ground truth masks
│       ├── seismic_slice_*.tif  # Input seismic images
│       └── *.png               # Ground truth annotations
└── README.md               # This documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/runabh1/oil-2.git
   cd oil-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app/dashboard.py
   ```

4. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Select seismic files from the sidebar dropdown
   - View AI predictions in real-time

## 🔬 Technical Details

### Geological Feature Classes

The AI model identifies and segments the following geological features:

| Class | Feature Type | Description |
|-------|-------------|-------------|
| 0 | Background/Noise | Non-geological areas and noise |
| 1 | Fault Type A | Primary fault systems |
| 2 | Fault Type B | Secondary fault systems |
| 3 | Horizon A | Primary layer boundaries |
| 4 | Horizon B | Secondary layer boundaries |
| 5 | Salt Dome | Salt diapir structures |
| 6 | Channel | Ancient river channels |
| 7 | Reservoir | Potential hydrocarbon zones |
| 8 | Unconformity | Erosion surfaces |
| 9 | Other Features | Additional geological structures |

### U-Net Architecture

```
Input Layer: 128×128×1 (grayscale seismic image)
    ↓
Encoder Path:
├── Conv Block 1: 64 filters → MaxPool + Dropout(0.1)
├── Conv Block 2: 128 filters → MaxPool + Dropout(0.1)
├── Conv Block 3: 256 filters → MaxPool + Dropout(0.2)
└── Bottleneck: 512 filters + Dropout(0.3)
    ↓
Decoder Path:
├── Upsample + Skip Connection → Conv Block: 256 filters
├── Upsample + Skip Connection → Conv Block: 128 filters
├── Upsample + Skip Connection → Conv Block: 64 filters
└── Output: Conv2D(10, 1×1) + Softmax
```

### Data Processing Pipeline

1. **Input Loading**: TIFF seismic images loaded via rasterio
2. **Preprocessing**: Resize to 128×128, normalize to [0,1] range
3. **AI Inference**: U-Net processes image and outputs probability maps
4. **Post-processing**: Convert probabilities to class predictions
5. **Visualization**: Apply color mapping for feature display

## 📊 Performance Metrics

- **Processing Speed**: ~1 second per seismic slice
- **Model Size**: 88.4 MB (trained weights)
- **Input Format**: TIFF seismic images
- **Output Format**: Color-coded segmentation masks
- **Accuracy**: Real-time calculation against ground truth annotations

## 🎨 User Interface

### Dashboard Layout

The Streamlit application features a clean, professional interface with:

- **Sidebar**: File selection dropdown for choosing seismic slices
- **Main Area**: Three-column layout for comprehensive analysis
  - **Column 1**: Original seismic input (grayscale)
  - **Column 2**: Geologist's ground truth annotations (colored)
  - **Column 3**: AI predicted segmentation (colored)
- **Metrics Panel**: Real-time accuracy and feature detection statistics
- **Technical Details**: Expandable section with model information

### Color Scheme

Each geological feature class is assigned a distinct color from the Spectral colormap:
- **Red**: Faults and discontinuities
- **Orange**: Horizons and layer boundaries
- **Yellow**: Salt domes and salt structures
- **Green**: Channels and fluvial features
- **Blue**: Reservoir zones and potential targets
- **Purple**: Unconformities and erosion surfaces

## 🔧 Configuration

### Model Parameters
- **Input Shape**: (128, 128, 1)
- **Number of Classes**: 10
- **Model Architecture**: U-Net with skip connections
- **Activation**: ReLU for hidden layers, Softmax for output

### File Paths
- **Model Weights**: `models/trained_interpai_unet_model.h5`
- **Test Data**: `data/test_samples/`
- **Seismic Images**: `seismic_slice_*.tif`
- **Ground Truth**: `*.png` (matching numbered files)

## 🌍 Real-World Applications

### Oil & Gas Industry Impact

**Traditional Workflow:**
- Manual seismic interpretation by geologists
- Hours to days per seismic section
- Subjective interpretation variability
- High labor costs and time investment

**With InterpAI:**
- Automated feature detection in seconds
- Consistent interpretation across datasets
- Reduced manual labor requirements
- Faster exploration decision-making

### Use Cases

1. **Exploration**: Rapid identification of potential drilling targets
2. **Reservoir Characterization**: Automated mapping of geological features
3. **Risk Assessment**: Consistent fault and horizon detection
4. **Training**: Educational tool for geoscience students
5. **Research**: Benchmark for seismic interpretation algorithms

## 📈 Business Value

- **⚡ Speed**: 100x faster than manual interpretation
- **🎯 Consistency**: Eliminates interpreter subjectivity
- **💰 Cost Reduction**: Reduces manual labor requirements
- **📊 Scalability**: Process large seismic datasets efficiently
- **🔬 Innovation**: Demonstrates AI potential in E&P industry

## 🛠️ Development

### Adding New Features

To extend the application:

1. **New Geological Classes**: Modify the `CLASSES` parameter and retrain the model
2. **Different Input Sizes**: Update the `IMG_SIZE` parameter and model architecture
3. **Additional Metrics**: Extend the metrics calculation in `dashboard.py`
4. **New Visualizations**: Add custom plotting functions in `prediction_service.py`

### Model Retraining

To train on new data:

1. Prepare seismic images and corresponding ground truth masks
2. Modify the U-Net architecture if needed
3. Implement training pipeline with TensorFlow/Keras
4. Save trained weights to `models/` directory
5. Update model loading in `dashboard.py`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:

- Bug reports and feature requests
- Code improvements and optimizations
- Additional geological feature classes
- Enhanced visualization options
- Documentation improvements



## 🙏 Acknowledgments

- **TensorFlow Team**: For the deep learning framework
- **Streamlit Team**: For the web application framework
- **Geoscience Community**: For seismic interpretation methodologies
- **Open Source Contributors**: For the various Python libraries used

## 📞 Contact

- **Repository**: [https://github.com/runabh1/oil-2](https://github.com/runabh1/oil-2)
- **Issues**: [GitHub Issues](https://github.com/runabh1/oil-2/issues)

---

**InterpAI** - Revolutionizing seismic interpretation through artificial intelligence 🌊⚡

*Built with ❤️ for the geoscience community*
