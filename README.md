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

# Overview
- [Quickstart Installation](#quickstart-installation)
- [Location of pachyderm pipelines](#location-of-pachyderm-pipelines)
- [Detailed Installation Steps](#detailed-installation-steps)
- [Bring your own documents](#bring-your-own-documents)
- [Bulid your own containers](#bulid-your-own-containers)
- [Bring your own Huggingface model](#bring-your-own-huggingface-model)
- [Bring your own Sentence Transformer model](#bring-your-own-sentence-transformer-model)

# Quickstart Installation

* Create new notebook on Houston using the `pdk-llm-rag-demo` template, you can select one gpu or no gpu.
* In your `shared_nb/01 - Users/<USER_NAME>` create a terminal and run`git clone   https://interactivetech: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@github.com/interactivetech/pdk-llm-rag-demo-test-.git`
* 

# Location of pachyderm pipelines:
* [deploy-rag](http://mldm-pachyderm.us.rdlabs.hpecorp.net/lineage/deploy-rag)
* [deploy-rag-finetune](http://mldm-pachyderm.us.rdlabs.hpecorp.net/lineage/deploy-rag-finetune)

## Notebooks to Run
* Run `Deploy RAG with PDK.pynb` to deploy a RAG system using a pretrained LLM
* Run `Finetune and Deploy RAG with PDK.ipynb` to both finetune an LLM and deploy a finetuned model.


# Detailed Installation Steps
We will show how to (in detail) setup this repo and demo for a new environment. We are assuming to create shared directories for storing model files:
* `/nvmefs1/test_user` and `/nvmefs1/test_user/cache`
Please modify this to your respective environment.

## Steps to Complete:
* Install Determined Notebook template to use all the required python libraries to run
* Create Determined notebook
* Setup directory of pretrained models and vectordb
* Modify `Deploy RAG with PDK.ipynb` notebook

Make sure you have a PDK deployment with a shared file directory `/nvmefs1/test_user` and `/nvmefs1/test_user/cache`

Create directory: `mkdir -p /nvmefs1/test_user`

### Review Determined Notebook template: pdk-llm-nb-env-houston.yaml


This is a configured template that mounts the host `/nvmefs1` directory to the determined notebook and training job.
You do not need to modify this file if you are running on houston.

### Install Determined Notebook template to use all the required python libraries to run
NOTE: You do not need to run this step on houston, as the notebook template `pdk-llm-rag-demo` is already accessible
change directory to the directory of this project, and run the command 

`det template create pdk-llm-rag-env env_files/pdk-llm-nb-env-houston.yaml`

next, create Notebook with No GPUs, or One GPU

Make sure you have a PDK deployment with a shared file directory

mkdir /run/determined/workdir/shared_fs/cache

## Create Determined notebook

Create a notebook using the `pdk-llm-rag-demo` template.

## Setup directory of pretrained models and vectordb
Make sure you have a PDK deployment with a shared file directory

`mkdir -p /nvmefs1/test_user/cache`

For the add_to_vector pipeline (defined in Deploy RAG with PDK.ipynb) create a directory that will store the persistent vector database

`mkdir -p /nvmefs1/test_user/cache/rag_db/`

Also we will need to cache the embedding model to vectorize our data into ChromaDB

` mkdir -p  /nvmefs1/test_user/cache/vector_model/all-MiniLM-L6-v2`

We will need to cache our LLM Model and Tokenizer

` mkdir -p  /nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer`

To prevent any interruption downloading, we will create a separate cache folder when first downloading the model
(We can delete this after successfully saving)
` mkdir -p  /nvmefs1/test_user/cache/model_cache/mistral_7b_model_tokenizer`

Finally, create a directory for the titanml cache:

`mkdir -p /nvmefs1/test_user/cache/titanml_cache`

Code to run to download vector db

`env/Download_Vector_Embedding.ipynb`

Code to run to download mistral 7B model

`env/Download_and_cache_Mistral_7B_model.ipynb`

## Get IPs to deploy RAG Application.

We need two IP Addresses that will allocate on the Houston Kubernetes Cluster. One IP will be used to deploy the TitanML API Service, and the user will deploy the user interface.



### Get the first IP
you will need two dedicated IPs that can persist on the houston cluster. Here are the steps I recommend running to make sure you can get IPs to use for the cluster. 

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: jupyter1
  labels:
    name: jupyter1
spec:
  containers:
  - name: ubuntu
    image: ubuntu:latest
    command: ["/bin/sh", "-c"]
    args:
      - echo starting;
        apt-get update;
        apt-get install -y python3 python3-pip;
        pip install jupyterlab;
        jupyter lab --ip=0.0.0.0 --port=8080 --NotebookApp.token='' --NotebookApp.password='' --allow-root
    ports:
    - containerPort: 8080
      hostPort: 8080
EOF
```

Once this is running, then run the command to assign the next available IP

```bash
kubectl expose pod jupyter1 --port 8080 --target-port 8080 --type LoadBalancer
```

Then run this command:
```bash
kubectl get svc jupyter1
```

And see the output:
```bash
[andrew@mlds-mgmt ~]$ kubectl get svc jupyter1
NAME       TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)          AGE
jupyter1   LoadBalancer   10.43.186.18   10.182.1.51   8080:31685/TCP   5s
```

We see that the ip address `10.182.1.51` is allocated, so save this IP address for the TitanML deployment. 

### Get the Second IP for the User Interface Pod

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: jupyter2
  labels:
    name: jupyter2
spec:
  containers:
  - name: ubuntu
    image: ubuntu:latest
    command: ["/bin/sh", "-c"]
    args:
      - echo starting;
        apt-get update;
        apt-get install -y python3 python3-pip;
        pip install jupyterlab;
        jupyter lab --ip=0.0.0.0 --port=8080 --NotebookApp.token='' --NotebookApp.password='' --allow-root
    ports:
    - containerPort: 8080
      hostPort: 8080
EOF
```

Once this is running, then run the command to assign the next available IP

```bash
kubectl expose pod jupyter2 --port 8080 --target-port 8080 --type LoadBalancer
```

Then run this command:
```bash
kubectl get svc jupyter2
```

And see the output:
```bash
[andrew@mlds-mgmt ~]$ kubectl get svc jupyter2
NAME       TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)          AGE
jupyter2   LoadBalancer   10.43.131.94   10.182.1.54   8080:31934/TCP   4s
```
We see that the ip address `10.182.1.54` is allocated, so save this IP address for the TitanML deployment. 

So we will use `10.182.1.51` for TitanML deployment, and `10.182.1.54` for the user interface deployment. 

Clean up pods and svcs

```bash
kubectl delete pod/jupyter1 && kubectl delete pod/jupyter2 && kubectl delete svc/jupyter1 && kubectl delete svc/jupyter2
```

# Modify `Deploy RAG with PDK.ipynb` notebook

This notebook allows SE's to drive how to continuosly update a vector database with new documents.

## There are two data repos:
* code: this is the repo that has all your code for preprocessing, training, and deployment. Code can be shown in the `src/` folder
* data: this includes all the raw XML files that contain HPE press releases

## Overview of pipelines we will deploy:
* **process_xml**: This runs `src/py/process_xmls.py` script to extract the text from the raw xml files, and save them into `/pfs/out/hpe_press_releases.csv`
* **add_to_vector_db**: This runs `src/py/seed.py` that takes `hpe_press_releases.csv` as input and indexes it to the vector database. NOTE: We are persisting the vector db as a folder in the directory you created `/nvmefs1/test_user/cache/rag_db/`
* **deploy**: This runs a runner script `src/scripts/generate_titanml_and_ui_pod_check.sh`. This script deploys the LLM located at `/nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer` to TitanML. TitanML does some efficient optimization so that models only uses 8.4GB on a GPU. 

### Can leave the **process_xml** pipline as is. 
No need to modify, will support any environment

### Modify the **add_to_vector_db** pipline
We will need to modify the **add_to_vector_db** pipeline yaml definition.

In jupyter notebook cell, make sure you modify te --path-to-db to the correct location:
```yaml
transform:
    image: mendeza/python38_process:0.2
    cmd: 
        - '/bin/sh'
    stdin: 
    - 'python /pfs/code/src/py/seed.py --path_to_db /nvmefs1/test_user/cache/rag_db/
    --csv_path /pfs/process_xml/hpe_press_releases.csv
    --emb_model_path /run/determined/workdir/shared_fs/cache/vector_model/all-MiniLM-L6-v2'
    - 'echo "$(openssl rand -base64 12)" > /pfs/out/random_file.txt'
```

### Modify the **src/scripts/generate_titanml_and_ui_pod_check.sh** script

go to `src/scripts/deploy_app.sh` and modify several variables:
* `TITANML_POD_NAME`
* `TITANML_CACHE_HOST`
* `HOST_VOLUME`
* `TAKEOFF_MODEL_NAME`
* `DB_PATH`
* `API_HOST`
* `UI_IP`
* `EMB_PATH`
* `APP_PY_PATH`

so it aligns with the current location of your shared directory:


Here is an example
```bash
# Environment variables
ROOT_DIR=/pfs/code/src/scripts/ # ROOT_DIR is the directory where the scripts reside in /pfs

TITANML_POD_NAME=titanml-pod # TITANML_POD_NAME is the name of the titanml pod we are deploying

TITANML_CACHE_HOST=/nvmefs1/test_user/cache/titanml_cache # TITANML_CACHE_HOST is the directory of the cache titanml needs during deployment

HOST_VOLUME=/nvmefs1/ # HOST_VOLUME is the path to the root mounted directory

TAKEOFF_MODEL_NAME=/nvmefs1/andrew.mendez/mistral_instruct_model_and_tokenizer/ # TAKEOFF_MODEL_NAME is the local path of a huggingface model titanml will optimize and deploy

TAKEOFF_DEVICE=cuda # TAKEOFF_DEVICE specifys to use GPU Acceleration for TitanML

API_PORT=8080
API_HOST=10.182.1.48 # This should update with the IP you verified
UI_POD_NAME=ui-pod
UI_PORT=8080
DB_PATH=/nvmefs1/test_user/cache/rag_db/ # DB_PATH is the path to the chromadb vector database

UI_IP=10.182.1.5 0# This should update with the second IP you verified
CHROMA_CACHE_HOST=/nvmefs1/andrew.mendez/chromadb_cache

EMB_PATH=/nvmefs1/test_user/cache/vector_model/e5-base-v2 
# APP_PY_PATH is the python path used to the python script that implements the UI
# Use /nvmefs1/ if you want fast debugging
APP_PY_PATH="/nvmefs1/shared_nb/01 - Users/andrew.mendez/2024/pdk-llm-rag-demo-test-/src/py/app.py"
```

You can run this as is, but if you want to deploy the TitanML API and the UI App on different IPS, change the `API_HOST` and the `UI_IP`

# Bring your own documents

[ToDo]

# Bulid your own containers

* Docker container for notebok environment: mendeza/mistral-rag-env:0.0.11-pachctl
Can build a similar container using:
```bash
FROM determinedai/environments:cuda-11.3-pytorch-1.12-tf-2.11-gpu-mpi-0.27.1
RUN pip install transformers==4.36.0
RUN pip install peft accelerate bitsandbytes trl
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install einops
RUN curl -L https://github.com/pachyderm/pachyderm/releases/download/v2.8.2/pachctl_2.8.2_linux_amd64.tar.gz | sudo tar -xzv --strip-components=1 -C /usr/local/bin
```
* Docker container for TitanML Serving appllication: mendeza/takeoff-mistral:0.5
    * To build a new `mendeza/takeoff-mistral` look at this [repo](https://github.com/interactivetech/takeoff-community) that includes:
        * the [Dockerfile](https://github.com/interactivetech/takeoff-community/blob/main/Dockerfile) 
        * and [build_container.sh](https://github.com/interactivetech/takeoff-community/blob/main/build_container.sh)
* Docker container for User Interface Chainlit app: mendeza/mistral-llm-rag-ui:0.0.7
```bash
FROM determinedai/environments:cuda-11.3-pytorch-1.12-tf-2.11-gpu-mpi-0.27.1
RUN pip install transformers==4.36.0
RUN pip install peft accelerate bitsandbytes trl
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install einops chainlit==0.7.700 sentence_transformers==2.2.2 sentencepiece==0.1.99
RUN curl -L https://github.com/pachyderm/pachyderm/releases/download/v2.8.2/pachctl_2.8.2_linux_amd64.tar.gz | sudo tar -xzv --strip-components=1 -C /usr/local/bin
```
* Docker container for PyTorchTrial Mistal finetuning: mendeza/mistral-rag-env:0.0.11-pachctl
    * Can build the same container using the instructions from mendeza/mistral-rag-env:0.0.11-pachctl


# Bring your own Huggingface model

Create a new folder for the new LLM Model and Tokenizer

` mkdir -p  /nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer2`

To prevent any interruption downloading, we will create a separate cache folder when first downloading the model
(We can delete this after successfully saving)
` mkdir -p  /nvmefs1/test_user/cache/model_cache/mistral_7b_model_tokenizer2`

Follow the notebook `env/Download_and_cache_Mistral_7B_model.ipynb` and modify the path to save model at `/nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer2` and the cache_dir `  /nvmefs1/test_user/cache/model_cache/mistral_7b_model_tokenizer2`

You will need to modify the `TAKEOFF_MODEL_NAME` in `src/scripts/deploy_app.sh` that points to the new local HF model

# Bring your own Sentence Transformer model

Create a new folder for the new Embedding Model

` mkdir -p  /nvmefs1/test_user/cache/vector_model/all-MiniLM-L6-v22`

Follow the notebook `env/Download_Vector_Embedding.ipynb` and modify the path to save model at `/nvmefs1/test_user/cache/vector_model/all-MiniLM-L6-v22`

You will need to modify the `EMB_PATH` in `src/scripts/deploy_app.sh` that points to the new local HF model