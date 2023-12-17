1. Install the requirements
    
    ```bash
    conda create -n <env_name> python=3.9
    conda activate <env_name>
    conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    pip install -r requirements
    ```
    
2. Download the data.
    
    Please follow the instructions in the published link [https://github.com/snaredataset/snare](https://github.com/snaredataset/snare)
    
3. Download the GPT-2 checkpoint in hugging face link [https://huggingface.co/gpt2](https://huggingface.co/gpt2)
4. Prepare the raw_np floder using scripts/extract_clip_features_resnet.py
    
5. Run the train scripts
    
    ```bash
    python train.py train.model=<model_name>
    ```