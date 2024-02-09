FROM determinedai/environments:cuda-11.3-pytorch-1.10-deepspeed-0.8.3-gpu-0.26.4
RUN pip install bitsandbytes git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/accelerate.git trl xformers wandb datasets einops gradio sentencepiece ipywidgets
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
