# install_dependencies.py
# Alternative installation script for InterpAI dependencies

import subprocess
import sys
import os

def install_package(package):
    """Install a package with error handling"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🔧 Installing InterpAI Dependencies...")
    print("=" * 50)
    
    # Core packages (install in order)
    packages = [
        "numpy>=1.26.0,<2.0.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "rasterio>=1.3.0",
        "streamlit==1.50.0",
        "tensorflow-cpu==2.16.1"
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 50)
    
    if failed_packages:
        print("❌ Some packages failed to install:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\n🔄 Trying alternative installation methods...")
        
        # Try installing failed packages individually
        for package in failed_packages:
            print(f"\n🔄 Retrying {package}...")
            install_package(package)
    else:
        print("✅ All packages installed successfully!")
    
    print("\n🚀 Installation complete!")
    print("Run: streamlit run app/dashboard.py")

if __name__ == "__main__":
    main()
