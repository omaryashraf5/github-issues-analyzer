#!/usr/bin/env python3
"""
Setup script for GitHub Issues Analyzer
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = ['reports', 'data', 'cache']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created successfully")


def main():
    """Main setup function"""
    print("🚀 GitHub Issues Analyzer Setup")
    print("=" * 50)

    # Install requirements
    if not install_requirements():
        sys.exit(1)

    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  Warning: NLTK data download failed. Some features may not work properly.")

    # Create directories
    create_directories()

    print("\n🎉 Setup completed successfully!")
    print("\nTo run the analyzer:")
    print("  python main.py")
    print("\nFor help:")
    print("  python main.py --help")


if __name__ == "__main__":
    main()
