# Multilingual Support Microservice 

The Language Detection microservice detects the language of the user's query as well as the response from the first llm microservice. It then configures a translation prompt to convert the answer from the response language to the query language. This prompt is sent to the second llm microservice to generate the final answer. This ensures seamless, accurate communication across different languages in real time.

This README provides set-up instructions and comprehensive details regarding the multilingual support microservice.

---


## 🚀 1. Start Microservice with Docker

### 1.1 Setup Environment Variables

```bash
export HF_TOKEN=${your_hf_api_token}
export MULTILINGUAL_SUPPORT_ENDPOINT="http://${your_ip}:8081"
```

### 1.2 Build Docker Image

```bash
cd ../../../
docker build -t opea/multilingual-support:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/multilingual_support/Dockerfile .
```

To start a docker container -

### 1.3 Run Docker with CLI

```bash
docker run -d --name="multilingual-support" -p 8001:8001 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e MULTILINGUAL_SUPPORT=$MULTILINGUAL_SUPPORT -e HF_TOKEN=$HF_TOKEN opea/multilingual-support:latest
```

---

## ✅ 2. Invoke Multilingual Support Microservice

The Multilingual Support microservice exposes following API endpoints:

- Check Service Status

  ```bash
  curl http://localhost:8001/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
  ```

- Execute multilingual support process by providing query 

  ```bash
  curl -X POST -H "Content-Type: application/json" -d @- http://localhost:8001/v1/multilingual_support <<JSON_DATA
  {
    "text": "Hi. I am doing fine.",
    "prompt": "### You are a helpful, respectful, and honest assistant to help the user with questions. \
    Please refer to the search results obtained from the local knowledge base. \
    But be careful to not incorporate information that you think is not relevant to the question. \
    If you don't know the answer to a question, please don't share false information. \
    ### Search results:   \n
    ### Question: 你好。你好吗？ \n
    ### Answer:"
  }
  JSON_DATA
  ```
