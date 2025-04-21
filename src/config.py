import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / os.getenv('MODEL_CHECKPOINT_DIR', 'models')
OUTPUT_DIR = PROJECT_ROOT / os.getenv('OUTPUT_DIR', 'output')
PLOTS_DIR = PROJECT_ROOT / os.getenv('PLOTS_DIR', 'plots')

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model configurations
LAG_LLAMA_CONFIG = {
    "repo_id": "time-series-foundation-models/Lag-Llama",
    "model_file": "lag-llama.ckpt",
    "local_dir": str(MODEL_DIR)
}

# Training configurations
DEVICE = os.getenv('DEVICE', 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 100))

def get_hf_token():
    """Get HuggingFace token from environment variable"""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        raise ValueError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "Please copy .env.example to .env and set your token."
        )
    return token 