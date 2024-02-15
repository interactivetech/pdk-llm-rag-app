# Continuous Retrieval Augmentation Generation (RAG) with the HPE MLOPs Platform
 
Author: andrew.mendez@hpe.com

This is a proof of concept showing how developers can create a Retrieval Augmentation Generation (RAG) system using Pachyderm and Determined AI.
This is a unique RAG system sitting on top of the HPE MLOPs platform, which is a combination of Pachyderm and Determined.AI. A RAG system built on top of an MLOPs platform allowing developers to continuously update and deploy a RAG application as more data is ingested.
We also provide an example of how developers can automatically trigger finetuning an LLM on a instruction tuning dataset.

We use the following technologies to implement the RAG System:
* ChromaDB for the vector database
* Chainlit for the User Interface
* Mistral-7B-Instruct for the large language model (LLM)
* DeterminedAI for finetuning the LLM
* Pachyderm to manage dataset versioning and pipeline orchestration.

# Pre-requisite
* This Demo requires running on a GPU. We support running an A100 80GB GPU or a Tesla T4 16GB GPU.
* This demo currently only supports deployment and running on the Houston Cluster
* This Demo assumes you have Pachyderm and DeterminedAI installed on top of Kubernetes. A guide will be provided soon to show how to install Pachyderm and Kubernetes.
* If you have a machine with GPUs, you can install PDK using this guide: https://github.com/interactivetech/pdk-install
* [WIP,Coming soon] We will modify the installation steps to also support installation on a PDK GCP cluster

# Overview
- [Quickstart Installation](#quickstart-installation)
- [Location of pachyderm pipelines](#location-of-pachyderm-pipelines)
- [Notebooks included in this demo](#notebooks-included-in-this-demo)
- [Detailed Installation Steps](#detailed-installation-steps)
- [Bring your own documents](#bring-your-own-documents)
- [Bulid your own containers](#bulid-your-own-containers)
- [Bring your own Huggingface model](#bring-your-own-huggingface-model)
- [Bring your own Sentence Transformer model](#bring-your-own-sentence-transformer-model)

# Quickstart Installation

* Create new notebook on the Houston cluster using the `pdk-llm-rag-demo` template, you can select one gpu or no gpu.
* In your `shared_nb/01 - Users/<USER_NAME>` create a terminal and run`git clone ttps://github.com/interactivetech/pdk-llm-rag-demo-test-.git`

* Open the `Deploy RAG with PDK.pynb`, and it should run out-of-the-box.
* Note: The default to deploy the TitanML pod is using an A100 (using the taint `A100-MLDM`), if you want to change this to deploy to a T4, do the following:
    * go to `src/scripts/deploy_app.sh`
    GPU_DEVICE=A100-MLDM
    * update # GPU_DEVICE=Tesla-T4


# Location of pachyderm pipelines:
* [deploy-rag](http://mldm-pachyderm.us.rdlabs.hpecorp.net/lineage/deploy-rag)
* [deploy-rag-finetune](http://mldm-pachyderm.us.rdlabs.hpecorp.net/lineage/deploy-rag-finetune)

## Notebooks included in this demo
* Run `Deploy RAG with PDK.pynb` to deploy a RAG system using a pretrained LLM
* Run `Finetune and Deploy RAG with PDK.ipynb` to both finetune an LLM and deploy a finetuned model.


# Detailed Installation Steps
We will show how to (in detail) setup this repo and demo for a new environment. We are assuming to create shared directories for storing model files:
* `/nvmefs1/test_user` and `/nvmefs1/test_user/cache`
Please modify this to your respective environment.

## Setup Jupyter Notebook in DeterminedAI environment
* Install DeterminedAI Notebook template to use all the required python libraries to run
* Create DeterminedAI notebook
* Setup directory of pretrained models and vectordb
* Modify `Deploy RAG with PDK.ipynb` notebook

Open a terminal, and make sure you create these folders within a shared file directory `/nvmefs1/test_user` and `/nvmefs1/test_user/cache`

Example command to create directory: `mkdir -p /nvmefs1/test_user/cache`

### Review Determined Notebook template: pdk-llm-nb-env-houston.yaml

`env_files/pdk-llm-nb-env-houston.yaml` is a configured template to setup all the packages needed to run the notebook demos. This template mounts the host `/nvmefs1` directory to the determined notebook and training job. You do not need to modify this file if you are running on houston.

### Install Determined Notebook template to use all the required python libraries to run
NOTE: You do not need to run this step on houston, as the notebook template `pdk-llm-rag-demo` is already accessible

If in an new environment, change directory to the directory of this project, and run the command 

`det template create pdk-llm-rag-env env_files/pdk-llm-nb-env-houston.yaml`

next, create Notebook with No GPUs, or One GPU

Create a notebook using the `pdk-llm-rag-demo` template.


## Setup directory of pretrained models and vectordb
Make sure you have a PDK deployment with a shared file directory

`mkdir -p /nvmefs1/test_user/cache`

Create a directory that will store the persistent vector database. This will be used for the add_to_vector pipeline (defined in Deploy RAG with PDK.ipynb) 

`mkdir -p /nvmefs1/test_user/cache/rag_db/`
We need to create a folder for  ChromaDB cache:

`mkdir -p /nvmefs1/test_user/cache/chromadb_cache `

We need to cache the embedding model to vectorize our data into ChromaDB

` mkdir -p  /nvmefs1/test_user/cache/vector_model/all-MiniLM-L6-v2`

We need to cache our LLM Model and Tokenizer

` mkdir -p  /nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer`

To prevent any interruption downloading, we will create a separate cache folder when first downloading the model
(We can delete this after successfully saving)
` mkdir -p  /nvmefs1/test_user/cache/model_cache/mistral_7b_model_tokenizer`

Finally, create a directory for the titanml cache:

`mkdir -p /nvmefs1/test_user/cache/titanml_cache`

Run the code in this notebook `env/Download_Vector_Embedding.ipynb` to download the embedding model to `/nvmefs1/test_user/cache/vector_model/all-MiniLM-L6-v2` 

Run the code in this notebook `env/Download_and_cache_Mistral_7B_model.ipynb` to run to download mistral 7B model t `/nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer` 


## Get IPs to deploy RAG Application.

We need two IP Addresses that will allocate on the Houston Kubernetes Cluster. One IP will be used to deploy the TitanML API Service, and the user will deploy the user interface.


### Get the first IP
you will need two dedicated IPs that can persist on the houston cluster. Here are the steps I recommend running to make sure you can get IPs to use for the cluster. 

Create a temporary pod on Houston Cluster:
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

Once this pod is running, run the command to assign the next available IP on the houston cluster

```bash
kubectl expose pod jupyter1 --port 8080 --target-port 8080 --type LoadBalancer
```

Then run this command to see what IP was allocated:
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

Create another temporary pod on Houston Cluster:

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

Clean up pods and svcs by running the command:

```bash
kubectl delete pod/jupyter1 && kubectl delete pod/jupyter2 && kubectl delete svc/jupyter1 && kubectl delete svc/jupyter2
```

# Modify `Deploy RAG with PDK.ipynb` notebook

This notebook allows SE's to drive how to continuosly update a vector database with new documents.

You will need to modify the notebook if you have a custom directory that is not `/nvmefs1/test_user/cache/`

Background: There are two data repos:

* code: this is the repo that has all your code for preprocessing, training, and deployment. Code can be shown in the `src/` folder
* data: this includes all the raw XML files that contain HPE press releases

Background: Overview of pipelines we will deploy:

* **process_xml**: This runs `src/py/process_xmls.py` script to extract the text from the raw xml files, and save them into `/pfs/out/hpe_press_releases.csv`
* **add_to_vector_db**: This runs `src/py/seed.py` that takes `hpe_press_releases.csv` as input and indexes it to the vector database. NOTE: We are persisting the vector db as a folder in the directory you created `/nvmefs1/test_user/cache/rag_db/`
* **deploy**: This runs a runner script `src/scripts/generate_titanml_and_ui_pod_check.sh`. This script deploys the LLM located at `/nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer` to TitanML. TitanML does some efficient optimization so that models only uses 8.4GB on a GPU. 


### Modify the **add_to_vector_db** pipline
Can leave the **process_xml** pipline as is. No need to modify, will support any environment

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


Here is an example values that work assuming you created and downloaded all the necessary files in `/nvmefs1/test_user/cache/` 
```bash
# Environment variables
ROOT_DIR=/pfs/code/src/scripts/ # ROOT_DIR is the directory where the scripts reside in /pfs

TITANML_POD_NAME=titanml-pod # TITANML_POD_NAME is the name of the titanml pod we are deploying

TITANML_CACHE_HOST=/nvmefs1/test_user/cache/titanml_cache # TITANML_CACHE_HOST is the directory of the cache titanml needs during deployment

HOST_VOLUME=/nvmefs1/ # HOST_VOLUME is the path to the root mounted directory

TAKEOFF_MODEL_NAME=/nvmefs1/test_user/cache/model/mistral_7b_model_tokenizer # TAKEOFF_MODEL_NAME is the local path of a huggingface model titanml will optimize and deploy

TAKEOFF_DEVICE=cuda # TAKEOFF_DEVICE specifys to use GPU Acceleration for TitanML

API_PORT=8080
API_HOST=10.182.1.48 # This should update with the IP you verified
UI_POD_NAME=ui-pod
UI_PORT=8080
DB_PATH=/nvmefs1/test_user/cache/rag_db/ # DB_PATH is the path to the chromadb vector database

UI_IP=10.182.1.5 0# This should update with the second IP you verified
CHROMA_CACHE_HOST= /nvmefs1/test_user/cache/chromadb_cache

EMB_PATH=/nvmefs1/test_user/cache/vector_model/e5-base-v2 
# APP_PY_PATH is the python path used to the python script that implements the UI
# Use /nvmefs1/ if you want fast debugging
APP_PY_PATH="/nvmefs1/shared_nb/01 - Users/andrew.mendez/2024/pdk-llm-rag-demo-test-/src/py/app.py"
```

You can run this as is, but if you want to deploy the TitanML API and the UI App on different IPS, change all the above values.

# Bring your own documents

Follow the [Bring_your_own_data.md](Bring_your_own_data.md)

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