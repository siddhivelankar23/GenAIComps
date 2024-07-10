# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from typing import Union

from langchain_community.embeddings import HuggingFaceHubEmbeddings
from embeddings.BridgeTowerEmbeddings import BridgeTowerEmbeddings
from langsmith import traceable

from comps import (
    EmbedDoc1024,
    ServiceType,
    TextDoc,
    ImageDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

@register_microservice(
    name="opea_service@embedding_multimodal",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=Union[TextDoc, ImageDoc],  # Updated to accept either TextDoc or ImageDoc
    output_datatype=EmbedDoc1024,
)
@traceable(run_type="embedding")
@register_statistics(names=["opea_service@embedding_multimodal"])

def embedding(input: Union[TextDoc, ImageDoc]) -> EmbedDoc1024:
    start = time.time()
    
    if isinstance(input, TextDoc):
        # Handle text input
        embed_vector = BridgeTowerEmbeddings.embed_query(input.text)
        res = EmbedDoc1024(text=input.text, embedding=embed_vector)
    elif isinstance(input, ImageDoc):
        # Handle image input
        embed_vector = BridgeTowerEmbeddings.embed_image_text_pairs(input.image_path)  # Adjust
        res = EmbedDoc1024(text=input.image_path, embedding=embed_vector)
        

    statistics_dict["opea_service@embedding_multimodal"].append_latency(time.time() - start, None)
    return res


if __name__ == "__main__":
    tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT", "http://localhost:8080")
    embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    print("TEI Gaudi Embedding initialized.")
    opea_microservices["opea_service@embedding_multimodal"].start()
