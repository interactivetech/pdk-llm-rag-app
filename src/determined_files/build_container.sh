DOCKER_IMG_NAME=mendeza/mistral-rag-env:0.0.1
echo "Building $DOCKER_IMG_NAME..."
docker build . -t $DOCKER_IMG_NAME --no-cache
echo "Done! Pushing image $DOCKER_IMG_NAME..."
docker push $DOCKER_IMG_NAME
echo "Done!"