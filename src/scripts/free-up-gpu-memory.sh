#!/bin/bash

POD_NAME="titanml-pod"
NAMESPACE="pachyderm"

install -o root -g root -m 0755 /mnt/efs/shared_fs/determined/kubectl /usr/local/bin/kubectl

if kubectl get pod -n "$NAMESPACE" "$POD_NAME" --ignore-not-found --output name | grep -q "$POD_NAME"; then
    echo "Deleting existing pod: $POD_NAME"
    kubectl delete pod -n "$NAMESPACE" "$POD_NAME"
else
    echo "Pod $POD_NAME does not exist."
fi