#installing lag-llama
#skips this step if the repository is already cloned
if not os.path.exists('lag-llama'):
#clones the lag-llama repository
    !git clone https://github.com/time-series-foundation-models/lag-llama.git
    print ("repository cloned successfully")
else:
    print ("lag-llama repository already cloned")
#installs the requirements
%pip install -r lag-llama/requirements.txt --no-deps
#first ensure you're logged in to huggingface
!huggingface-cli login --token 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
#skips this step if the model is already downloaded
if not os.path.exists('lag-llama.ckpt'):
    try:
       # Download using the Hugging Face hub library 
        file = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
            local_dir=".",
            token=True  
        )
        print(f"Downloaded {file} successfully")
    except Exception as e:
        print(f"Error downloading model: {e}")
else:
    print ("model checkpoint already downloaded")