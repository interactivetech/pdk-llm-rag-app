# Continuous Retrieval Augmentation Generation (RAG) with the HPE MLOPs Platform
 
Author: andrew.mendez@hpe.com

This is a proof of concept showing how developers can create a Retrieval Augmentation Generation (RAG) system using Pachyderm and Determined AI.
This is a unique RAG system sitting on top of an MLOPs platform, allowing developers to continuously update and deploy a RAG application as more data is ingested.
We also provide an example of how developers can automatically trigger finetuning an LLM on a instruction tuning dataset.

We use the following stack:
* ChromaDB for the vector database
* Chainlit for the User Interface
* Mistral 7B Instruct for the large language model
* Determined for finetuning the Mistral Model
* Pachyderm to manage dataset versioning and pipeline orchestration.

# Pre-requisite
* This Demo requires running with an A100 80GB GPU.
* This demo currently only supports deployment and running on the Houston Cluster
* This Demo assumes you have pachyderm and determined installed on top of kubernetes. A guide will be provided soon to show how to install pachyderm and kubernetes.
* If you have a machine with GPUs, you can install PDK using this guide: https://github.com/interactivetech/pdk-install
* [Coming soon] We will modify the installation steps to also support installation on a PDK GCP cluster

# How to Run
* Run `Deploy RAG with PDK.pynb` to deploy a RAG system using a pretrained LLM
* Run `Finetune and Deploy RAG with PDK.ipynb` to both finetune an LLM and deploy a finetuned model.

# Installation Steps
Make sure you have a PDK deployment with a shared file directory `/nvmefs1/test_user`

Create directory: `mkdir -p /nvmefs1/test_user`

# Update 

## Install Determined Notebook template to use all the required python libraries to run
change directory to the directory of this project, and run the command 

`det template create pdk-llm-rag-env env/pdk-llm-nb-env.yaml`

next, create Notebook with No GPUs, or One GPU

Make sure you have a PDK deployment with a shared file directory

mkdir /run/determined/workdir/shared_fs/cache