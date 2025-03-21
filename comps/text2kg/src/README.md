# Text to knowledge graph microservice

Text to Knowledge Graph (text2kg) Microservice is a specialized service designed to extract structured knowledge graphs from unstructured text data. Built using an encoder-decoder architecture, it leverages advanced Large Language Models (LLMs) to identify entities and relationships within text documents, converting them into meaningful graph triplets. This microservices uses the neo4j database to store the data and tgi microservice for using the llm.

# 🚀 1. Start microservice with Docker (Option 1)

## A. Start individual microservices using docker cli (Option A)

### Install Requirements

```bash
 pip install -r requirements.txt
```

### 1. TGI 

#### a. Start the TGI microservice
```bash

export TGI_PORT=8008
export HF_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID=${LLM_MODEL_ID:-"HuggingFaceH4/zephyr-7b-alpha"}
export LLM_ENDPOINT_PORT=${LLM_ENDPOINT_PORT:-"9001"}

export TGI_PORT=8008
export PYTHONPATH="/home/user/"

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

#### b. Configure the username, password, dbname 

```bash
export NEO4J_AUTH=neo4j/password
export NEO4J_PLUGINS=\[\"apoc\"\]
export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4j_password"}
export NEO4J_PORT1={$NEO4J_PORT1:-7474}:7474
export NEO4J_PORT2={$NEO4J_PORT2:-7687}:7687
```

Export temporary directory and make sure the files that need to be queried are in this temporary directory.

Export relational variables based on your text. For example -
```bash
export TEMP_DIR=$(pwd)
export ENTITIES="PERSON,PLACE,ORGANIZATION"
export RELATIONS="HAS,PART_OF,WORKED_ON,WORKED_WITH,WORKED_AT"
export VALIDATION_SCHEMA='{
    "PERSON": ["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"],
    "PLACE": ["HAS", "PART_OF", "WORKED_AT"],
    "ORGANIZATION": ["HAS", "PART_OF", "WORKED_WITH"]
}'
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

```bash
cd comps/text2kg/src/
```

Build the text2kg docker image
```bash
docker build -f Dockerfile -t opea/text2kg:latest ../../../
```

Launch the docker container
```bash
docker run -i -t --net=host --ipc=host -p 8090 opea/text2kg:latest -v data:/home/user/comps/text2kg/src/data /bin/bash
```


## B. Start text2kg and dependent microservices with docker-compose (Option B)
```bash
cd comps/text2kg/deployment/docker_compose/
```
Export service name and log path
```bash
export service_name="text2kg"
export LOG_PATH=$PWD
```
Export NEO4J variables - refer to section 1.A.2.b.

Launch using the following command to run on cpu
```bash
docker compose -f compose.yaml -f custom-override.yml up ${service_name}  -d > ${LOG_PATH}/start_services_with_compose.log
```
Launch using  the following command to run on gaudi
```bash
docker compose -f compose.yaml up ${service_name}  -d > ${LOG_PATH}/start_services_with_compose.log
```
# 🚀 2. Start microservice with Python (Option 2)


## Install Requirements

```bash
 pip install -r requirements.txt
```

## Start tgi and neo4j mircoservices 
Refer to sections 1.A.1. and 1.A.2.

## Start text2kg microservice using python script 

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
