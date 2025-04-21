import os
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download
from config import MODEL_DIR, LAG_LLAMA_CONFIG, get_hf_token

def clone_repository():
    """Clone the Lag-LLaMA repository if not already present"""
    repo_path = MODEL_DIR / "lag-llama"
    if not repo_path.exists():
        print("Cloning Lag-LLaMA repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/time-series-foundation-models/lag-llama.git",
            str(repo_path)
        ], check=True)
        print("Repository cloned successfully")
    else:
        print("Lag-LLaMA repository already exists")
    return repo_path

def install_requirements(repo_path):
    """Install Lag-LLaMA requirements"""
    print("Installing Lag-LLaMA requirements...")
    requirements_path = repo_path / "requirements.txt"
    subprocess.run([
        "pip", "install", "-r", str(requirements_path), "--no-deps"
    ], check=True)
    print("Requirements installed successfully")

def download_model():
    """Download the Lag-LLaMA model checkpoint"""
    checkpoint_path = MODEL_DIR / LAG_LLAMA_CONFIG["model_file"]
    if not checkpoint_path.exists():
        print("Downloading Lag-LLaMA model checkpoint...")
        try:
            file = hf_hub_download(
                repo_id=LAG_LLAMA_CONFIG["repo_id"],
                filename=LAG_LLAMA_CONFIG["model_file"],
                local_dir=LAG_LLAMA_CONFIG["local_dir"],
                token=get_hf_token()
            )
            print(f"Model checkpoint downloaded successfully to {file}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print("Model checkpoint already exists")
    return checkpoint_path

def main():
    """Main installation function"""
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Clone repository
        repo_path = clone_repository()
        
        # Install requirements
        install_requirements(repo_path)
        
        # Download model
        checkpoint_path = download_model()
        
        print("\nLag-LLaMA installation completed successfully!")
        print(f"Repository: {repo_path}")
        print(f"Model checkpoint: {checkpoint_path}")
        
    except Exception as e:
        print(f"\nError during installation: {e}")
        raise

if __name__ == "__main__":
    main() 