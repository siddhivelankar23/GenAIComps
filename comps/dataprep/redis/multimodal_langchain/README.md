# Dataprep Microservice for Multimodal Data with Redis

This dataprep microservice accepts videos (mp4 files) from the user and ingests data into Redis vectorstore with the help of transcripts and captions.

For videos without audio or recognizable speech, LVM is used to generate captions for video frames. To leverage LVM, please refer to this [readme](../../../lvms/README.md) to start the LVM microservice first before starting this microservice.

# 🚀1. Start Microservice with Python（Option 1）

## 1.1 Install Requirements

```bash
apt update
apt install default-jre

# Install ffmpeg static build
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
mkdir ffmpeg-git-amd64-static
tar -xvf ffmpeg-git-amd64-static.tar.xz -C ffmpeg-git-amd64-static --strip-components 1
export PATH=$(pwd)/ffmpeg-git-amd64-static:$PATH
cp $(pwd)/ffmpeg-git-amd64-static/ffmpeg /usr/local/bin/

pip install -r requirements.txt
```

## 1.2 Start Redis Stack Server

Please refer to this [readme](../../../vectorstores/langchain/redis/README.md).

## 1.3 Setup Environment Variables

```bash
export REDIS_URL="redis://${your_ip}:6379"
export INDEX_NAME=${your_index_name}
export PYTHONPATH=${path_to_comps}
```

## 1.4 Start LVM Microservice

Please refer to this [readme](../../../lvms/README.md) to start the LVM microservice.

After LVM is up, set up environment variables.

```bash
export LVM_ENDPOINT="http://localhost:9399/v1/lvm"
```

## 1.5 Start Document Preparation Microservice for Redis with Python Script

Start document preparation microservice for Redis with below command.

```bash
python prepare_videodoc_redis.py
```

# 🚀2. Start Microservice with Docker (Option 2)

## 2.1 Start Redis Stack Server

Please refer to this [readme](../../../vectorstores/langchain/redis/README.md).

## 2.2 Setup Environment Variables

```bash
export EMBEDDING_MODEL_ID="BridgeTower/bridgetower-large-itm-mlm-itc"
export LVM_ENDPOINT="http://${your_ip}:9399/v1/lvm"
export REDIS_URL="redis://${your_ip}:6379"
export INDEX_NAME=${your_index_name}
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
```

## 2.3 Build Docker Image

```bash
cd ../../../../../
docker build -t opea/dataprep-redis:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/redis/multimodal_langchain/docker/Dockerfile .
```

## 2.4 Run Docker with CLI (Option A)

```bash
docker run -d --name="dataprep-redis-server" -p 6007:6007 --runtime=runc --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e REDIS_URL=$REDIS_URL -e INDEX_NAME=$INDEX_NAME -e LVM_ENDPOINT=$LVM_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN opea/dataprep-redis:latest
```

## 2.5 Run with Docker Compose (Option B - deprecated, will move to genAIExample in future)

```bash
cd comps/dataprep/redis/multimodal_langchain/docker
docker compose -f docker-compose-dataprep-redis.yaml up -d
```

# 🚀3. Status Microservice

```bash
docker container logs -f dataprep-redis-server
```

# 🚀4. Consume Microservice

## 4.1 Consume videos_with_transcripts API

Once document preparation microservice for Redis is started, user can use below command to invoke the microservice to convert videos and their transcripts to embeddings and save to the database.

Make sure the file path after `files=@` is correct.

- Single video-transcript pair upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video1.vtt" \
    http://dataprep-redis-service:6007/v1/dataprep/videos_with_transcripts
```

- Multiple video-transcript pair upload
```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video1.vtt" \
    -F "files=@./video2.mp4" \
    -F "files=@./video2.vtt" \
    http://dataprep-redis-service:6007/v1/dataprep/videos_with_transcripts
```

## 4.2 Consume generate_transcripts API

If transcripts are not available for videos, transcripts will be extracted from them. The user can use below command to invoke the microservice to convert videos and their extracted transcripts to embeddings and save to the database.

- Single video upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    http://dataprep-redis-service:6007/v1/dataprep/generate_transcripts
```

- Multiple video upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video2.mp4" \
    http://dataprep-redis-service:6007/v1/dataprep/generate_transcripts
```

## 4.3 Consume generate_captions API

If uploaded videos lack audio or recognizable speech, captions will be generated for frames using LVM. The user can use below command to invoke the microservice to convert videos and generated captions to embeddings and save to the database.

- Single video upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    http://dataprep-redis-service:6007/v1/dataprep/generate_captions
```

- Multiple video upload

```bash
curl -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./video1.mp4" \
    -F "files=@./video2.mp4" \
    http://dataprep-redis-service:6007/v1/dataprep/generate_captions
```

## 4.4 Consume get_videos API

To get names of uploaded videos, use the following command.

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    http://dataprep-redis-service:6007/v1/dataprep/get_videos
```

## 4.5 Consume delete_videos API

To delete uploaded videos and clear the database, use the following command.

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    http://dataprep-redis-service:6007/v1/dataprep/delete_videos
```
