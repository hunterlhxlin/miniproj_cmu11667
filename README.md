# 11-667 Homework 6: Miniproject

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual
  machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1Tw_klO84R9G7CZ3cINAKgy4BfdNm-8dlnRXSBIVD_3A/edit?usp=sharing) 

### Python environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`
2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
```bash
conda create -n cmu-llms-hw6 python=3.11
conda activate cmu-llms-hw6
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
pip install -r requirements.txt
pip install wandb
pip install -U "huggingface_hub[cli]"
```
3. Run `wandb login` to finish setting up weights & biases for experiment tracking (you will need to have a [weights & biases account](https://wandb.ai/login)).  

4. Run `huggingface-cli login` to allow downloading and uploading to Huggingface hub. 


### Preparing the data
Due to the fact that `lmsys/chatbot_arena_conversations` has only one split, plz refer to `data_split.py` to ensure the consistency of the val and test sets. Use a fixed rand seed. Also for simplicity, we filter out conversations that are not English.


### Training and Evaluation
Please refer to `train.py` and `evaluate.py`