# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository
Ensure your GitHub repository contains:
- ✅ `requirements.txt` (root level)
- ✅ `app/dashboard.py` (main application)
- ✅ `src/` directory with model files
- ✅ `models/trained_interpai_unet_model.h5`
- ✅ `data/test_samples/` with sample data

### Step 2: Deploy on Streamlit Cloud

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with your GitHub account
3. **Click**: "New app"
4. **Repository**: Select `runabh1/oil-2`
5. **Branch**: `main`
6. **Main file path**: `app/dashboard.py`
7. **App URL**: Choose your custom URL
8. **Click**: "Deploy!"

### Step 3: Monitor Deployment

The deployment process will:
1. Install dependencies from `requirements.txt`
2. Start the Streamlit application
3. Make it available at your custom URL

### 🔧 Troubleshooting Deployment Issues

#### Issue: TensorFlow Installation Failed
**Solution**: The updated `requirements.txt` now uses `tensorflow==2.20.0` which is compatible with Streamlit Cloud.

#### Issue: Module Import Errors
**Solution**: The updated `dashboard.py` now handles import errors gracefully and provides helpful error messages.

#### Issue: Model File Not Found
**Solution**: Ensure the model file is committed to the repository and accessible.

### 📊 Expected Deployment Time
- **First deployment**: 5-10 minutes
- **Subsequent updates**: 2-5 minutes

### 🌐 Access Your Deployed App
Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

### 🔄 Updating Your App
To update your deployed app:
1. Make changes to your code
2. Commit and push to GitHub
3. Streamlit Cloud will automatically redeploy

### 📞 Support
If deployment fails:
1. Check the deployment logs in Streamlit Cloud
2. Verify all files are present in the repository
3. Ensure `requirements.txt` has compatible versions
4. Check the troubleshooting section in `TROUBLESHOOTING.md`
