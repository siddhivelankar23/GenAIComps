# Text to knowledge graph microservice

Text to Knowledge Graph (text2kg) Microservice is a specialized service designed to extract structured knowledge graphs from unstructured text data. Built using an encoder-decoder architecture, it leverages advanced Large Language Models (LLMs) to identify entities and relationships within text documents, converting them into meaningful graph triplets. This microservices uses the neo4j database to store the data and tgi microservice for using the llm.

# 🚀 1. Start microservice with Docker (Option 1)

## A. Start individual microservices using docker cli (Option A)

### Install Requirements

```bash
 pip install -r requirements.txt
```

### Configure LLM Parameters based on the model selected.

```
export HF_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID=${LLM_MODEL_ID:-"HuggingFaceH4/zephyr-7b-alpha"}
export LLM_ENDPOINT_PORT=${LLM_ENDPOINT_PORT:-"9001"}

export SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
export OVERLAP=${OVERLAP:-"100"}
export MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
export TGI_PORT=8008
export PYTHONPATH="/home/user/"

export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4j_password"}
export NEO4J_PORT1={$NEO4J_PORT1:-7474}:7474
export NEO4J_PORT2={$NEO4J_PORT2:-7687}:7687
```


### 1. TGI 

#### a. Start the TGI microservice
```bash
export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export TGI_PORT=8008

docker run -d --name="text2graph-tgi-endpoint" --ipc=host -p $TGI_PORT:80 -v ./data:/data --shm-size 1g -e HF_TOKEN=${HF_TOKEN} -e model=${LLM_MODEL_ID} ghcr.io/huggingface/text-generation-inference:2.1.0 --model-id $LLM_MODEL_ID
```

#### b. Verify the TGI microservice

```bash
export your_ip=$(hostname -I | awk '{print $1}')
curl http://${your_ip}:${TGI_PORT}/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```

#### c. Setup Environment Variables to host TGI

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:${TGI_PORT}"
```
### 2. Neo4J
#### a. Download Neo4J image

```bash
docker pull neo4j:latest
```

#### b. Configure the username, password and dbname

```bash
export NEO4J_AUTH=neo4j/password
export NEO4J_PLUGINS=\[\"apoc\"\]
```

#### c. Run Neo4J service

Launch the database with the following docker command.

```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=\[\"apoc\"\]  \
    neo4j:latest
```

### 3. Text2kg

Build the text2kg docker image
```bash
docker build -f Dockerfile -t opea/text2kg:latest ../../../
```

Launch the docker container
```bash
docker run -i -t --net=host --ipc=host -p 8090 opea/text2kg:latest -v comps/text2kg/src/data:/home/user/comps/text2kg/src/data /bin/bash
```


## B. Start text2kg and dependent microservices with docker-compose (Option B)
```bash
comps/text2kg/deployment/docker_compose/
```
Export service name and log path
```bash
export service_name="text2kg"
export LOG_PATH=$PWD
```

Launch using  the following command to run on cpu
```bash
docker compose -f compose.yaml -f custom-override.yml up ${service_name}  -d > ${LOG_PATH}/start_services_with_compose.log
```
# 🚀 2. Start microservice with Docker (Option 2)


## Install Requirements

```bash
 pip install -r requirements.txt
```

## Start tgi and neo4j mircoservices - 
Refer to sections 1.A.1. and 1.A.2.

## Start text2kg microservice using python script - 

```bash
python3 comps/text2kg/src/opea_text2kg_microservice.py
```

# 3. Check the service using API endpoint

```bash
curl -X 'POST' \
  'http://localhost:8090/v1/text2kg?input_text=Who%20is%20paul%20graham%3F' \
  -H 'accept: application/json' \
  -d ''
```
