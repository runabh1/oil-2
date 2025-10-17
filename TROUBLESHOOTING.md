# TROUBLESHOOTING.md
# InterpAI Dependency Installation Troubleshooting Guide

## 🚨 Common Installation Issues

### 1. TensorFlow DLL Error
**Error**: `DLL load failed while importing _pywrap_tensorflow_internal`

**Solutions**:
```bash
# Option 1: Install Microsoft Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Option 2: Use alternative TensorFlow installation
pip uninstall tensorflow tensorflow-cpu -y
pip install tensorflow-cpu==2.16.1 --no-cache-dir

# Option 3: Install with conda (if available)
conda install tensorflow-cpu=2.16.1
```

### 2. Rasterio Installation Issues
**Error**: `Microsoft Visual C++ 14.0 is required`

**Solutions**:
```bash
# Option 1: Install pre-compiled wheel
pip install --only-binary=all rasterio

# Option 2: Use conda
conda install rasterio

# Option 3: Install GDAL first
pip install GDAL
pip install rasterio
```

### 3. OpenCV Installation Issues
**Error**: `Could not find a version that satisfies the requirement`

**Solutions**:
```bash
# Option 1: Install specific version
pip install opencv-python==4.8.0.76

# Option 2: Install headless version
pip install opencv-python-headless

# Option 3: Use conda
conda install opencv
```

## 🔧 Alternative Installation Methods

### Method 1: Use the Installation Script
```bash
python install_dependencies.py
```

### Method 2: Install Packages Individually
```bash
pip install numpy==1.26.4
pip install pillow==10.0.0
pip install matplotlib==3.7.0
pip install pandas==2.0.0
pip install opencv-python==4.8.0.76
pip install scikit-learn==1.3.0
pip install scikit-image==0.21.0
pip install rasterio==1.3.0
pip install streamlit==1.50.0
pip install tensorflow-cpu==2.16.1
```

### Method 3: Use Conda Environment
```bash
# Create new environment
conda create -n interpai python=3.11

# Activate environment
conda activate interpai

# Install packages
conda install numpy matplotlib pandas scikit-learn scikit-image
conda install -c conda-forge rasterio opencv streamlit
pip install tensorflow-cpu==2.16.1
```

## 🐛 Runtime Issues

### Issue: ModuleNotFoundError: No module named 'src'
**Solution**: Run from project root directory
```bash
cd C:\Users\aruna\OneDrive\Desktop\seismic-interpai-portfolio
streamlit run app/dashboard.py
```

### Issue: Model file not found
**Solution**: Ensure model file exists
```bash
# Check if model exists
ls models/trained_interpai_unet_model.h5

# If missing, download from repository
git pull origin main
```

### Issue: No TIFF files found
**Solution**: Check test data directory
```bash
# Verify test files exist
ls data/test_samples/*.tif

# If missing, the conversion script should have created them
python -c "import numpy as np; print('Files:', [f for f in os.listdir('data/test_samples') if f.endswith('.tif')])"
```

## 🔍 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.11
- **RAM**: 16GB
- **Storage**: 5GB free space
- **CPU**: Multi-core processor with AVX support

## 📞 Getting Help

If you continue to experience issues:

1. **Check Python Version**: `python --version`
2. **Check Pip Version**: `pip --version`
3. **Update Pip**: `python -m pip install --upgrade pip`
4. **Clear Cache**: `pip cache purge`
5. **Create Issue**: [GitHub Issues](https://github.com/runabh1/oil-2/issues)

## 🎯 Quick Test

After installation, test if everything works:
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit version:', st.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

If all commands succeed, you're ready to run the application!
