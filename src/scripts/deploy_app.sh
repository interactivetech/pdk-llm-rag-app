#!/bin/bash

# Environment variables
ROOT_DIR=/pfs/code/src/scripts/ # ROOT_DIR is the directory where the scripts reside in /pfs

TITANML_POD_NAME=titanml-pod # TITANML_POD_NAME is the name of the titanml pod we are deploying

TITANML_CACHE_HOST=/nvmefs1/test_user/cache/titanml_cache # TITANML_CACHE_HOST is the directory of the cache titanml needs during deployment

HOST_VOLUME=/nvmefs1/ # HOST_VOLUME is the path to the root mounted directory

TAKEOFF_MODEL_NAME=/nvmefs1/andrew.mendez/mistral_instruct_model_and_tokenizer/ # TAKEOFF_MODEL_NAME is the local path of a huggingface model titanml will optimize and deploy

TAKEOFF_DEVICE=cuda # TAKEOFF_DEVICE specifys to use GPU Acceleration for TitanML

API_PORT=8080
API_HOST=10.182.1.48
UI_POD_NAME=ui-pod
UI_PORT=8080
DB_PATH=/nvmefs1/test_user/cache/rag_db3 # DB_PATH is the path to the chromadb vector database

UI_IP=10.182.1.51
CHROMA_CACHE_HOST=/nvmefs1/andrew.mendez/chromadb_cache

# EMB_PATH=/nvmefs1/andrew.mendez/chromadb_cache/all-MiniLM-L6-v2
# EMB_PATH=/nvmefs1/test_user/cache/vector_model/e5-mistral-7b-instruct
EMB_PATH=/nvmefs1/test_user/cache/vector_model/e5-base-v2 
# APP_PY_PATH is the python path used to the python script that implements the UI
# Use /nvmefs1/ if you want fast debugging
APP_PY_PATH="/nvmefs1/shared_nb/01 - Users/andrew.mendez/2024/pdk-llm-rag-demo-test-/src/py/app.py"
# APP_PY_PATH="/pfs/out/app.py"

# GPU_DEVICE=Tesla-T4
GPU_DEVICE=A100-MLDM
# Install Kubectl command removed as it was commented out
install -o root -g root -m 0755 /nvmefs1/andrew.mendez/kubectl /usr/local/bin/kubectl

# Check if the TitanML pod exists
if kubectl get pod -n pachyderm "$TITANML_POD_NAME" --ignore-not-found --output name | grep -q "$TITANML_POD_NAME"; then
    echo "Pod $TITANML_POD_NAME exists."
else
    echo "Pod $TITANML_POD_NAME does not exist, creating..."
    
    sed -e "s|{{LOCAL_TITANML_CACHE}}|$TITANML_CACHE_HOST|g" \
       -e "s|{{LOCAL_VOLUME}}|$HOST_VOLUME|g" \
       -e "s|{{TAKEOFF_MODEL_NAME}}|$TAKEOFF_MODEL_NAME|g" \
       -e "s|{{API_PORT}}|$API_PORT|g" \
       -e "s|{{API_HOST}}|$API_HOST|g" \
       -e "s|{{TAKEOFF_DEVICE}}|$TAKEOFF_DEVICE|g" \
       -e "s|{{GPU_DEVICE}}|$GPU_DEVICE|g" \
       "$ROOT_DIR"/titanml-pod-template-v2.yaml > "$ROOT_DIR"/titanml-pod-runner.yaml
    kubectl apply -f "$ROOT_DIR"/titanml-pod-runner.yaml
    kubectl wait --for=condition=ready pod/titanml-pod

    echo "Done!"
fi

# Check if the UI pod exists
if kubectl get pod -n pachyderm "$UI_POD_NAME" --ignore-not-found --output name | grep -q "$UI_POD_NAME"; then
    echo "Pod $UI_POD_NAME exists, restarting..."
    kubectl delete pod -n pachyderm $UI_POD_NAME
    # The wait command for POD_NAME is adjusted to UI_POD_NAME for consistency
    kubectl wait --for condition=Ready=False --timeout=1h pod/$UI_POD_NAME
    echo "Restarted!"
else
    echo "Pod $UI_POD_NAME does not exist, creating..."
fi

sed -e "s|{{UI_PORT}}|$UI_PORT|g" \
   -e "s|{{DB_PATH}}|$DB_PATH|g" \
   -e "s|{{API_PORT}}|$API_PORT|g" \
   -e "s|{{API_HOST}}|$API_HOST|g" \
   -e "s|{{UI_IP}}|$UI_IP|g" \
   -e "s|{{LOCAL_CHROMA_CACHE}}|$CHROMA_CACHE_HOST|g" \
   -e "s|{{LOCAL_VOLUME}}|$HOST_VOLUME|g" \
   -e "s|{{APP_PY_PATH}}|\"$APP_PY_PATH\"|g" \
   -e "s|{{EMB_PATH}}|\"$EMB_PATH\"|g" \
   "$ROOT_DIR"/ui-pod-template-v2.yaml > "$ROOT_DIR"/ui-pod-runner.yaml
kubectl apply -f "$ROOT_DIR"/ui-pod-runner.yaml
kubectl wait --for=condition=ready pod/ui-pod

echo "Done!"