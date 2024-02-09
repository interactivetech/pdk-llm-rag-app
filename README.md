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
* This Demo assumes you have pachyderm and determined installed on top of kubernetes. A guide will be provided soon to show how to install pachyderm and kubernetes.

# How to Run
* Run `Deploy RAG with PDK.pynb` to deploy a RAG system using a pretrained LLM
* Run `Finetune and Deploy RAG with PDK.ipynb` to both finetune an LLM and deploy a finetuned model.