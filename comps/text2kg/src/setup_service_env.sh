# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#######################################################################
# Proxy
#######################################################################
export https_proxy=${https_proxy}
export http_proxy=${http_proxy}
export no_proxy=${no_proxy}
################################################################
# Configure LLM Parameters based on the model selected.
################################################################
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export HF_TOKEN=${HF_TOKEN}

export LLM_ID=${LLM_ID:-"HuggingFaceH4/zephyr-7b-alpha"}
export LLM_MODEL_ID=${LLM_MODEL_ID:-"HuggingFaceH4/zephyr-7b-alpha"}
export LLM_ENDPOINT_PORT=${LLM_ENDPOINT_PORT:-"9001"}

export SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
export OVERLAP=${OVERLAP:-"100"}
export MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
export TGI_PORT=8008
export PYTHONPATH="/home/user/"

export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4j_password"}
export NEO4J_PORT1={$NEO4J_PORT1:-8484}:8484
export NEO4J_PORT2={$NEO4J_PORT2:-8687}:8687


################################################################
### Echo env variables
################################################################
echo "Extractor details"
echo LLM_ID=${LLM_ID}
echo SPAN_LENGTH=${SPAN_LENGTH}
echo OVERLAP=${OVERLAP}
echo MAX_LENGTH=${MAX_LENGTH}
