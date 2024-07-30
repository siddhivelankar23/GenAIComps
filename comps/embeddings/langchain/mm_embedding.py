# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from typing import Union


from embeddings.BridgeTowerEmbeddings import BridgeTowerEmbeddings, MMEmbeddings
from langsmith import traceable

from comps import (
    EmbedDoc1024,
    ServiceType,
    TextDoc,
    ImageDoc,
    TextImageDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

MMDoc = Union[TextDoc, ImageDoc, TextImageDoc]

@register_microservice(
    name="opea_service@embedding_multimodal",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=MMDoc,  
    output_datatype=EmbedDoc1024,
)
@traceable(run_type="embedding")
@register_statistics(names=["opea_service@embedding_multimodal"])




def embedding(input: MMDoc) -> EmbedDoc1024:
    start = time.time()
    
    if isinstance(input, TextDoc):
        # Handle text input
        embed_vector = MMEmbeddings.embed_query(input.text)
        res = EmbedDoc1024(text=input.text, embedding=embed_vector)
    elif isinstance(input, ImageDoc):
        # Handle image input
        embed_vector = MMEmbeddings.embed_image(input.image_path)  
        res = EmbedDoc1024(text=input.image_path, embedding=embed_vector)
    elif isinstance(input, TextImageDoc):
        # Handle text + image input
        embed_vector = MMEmbeddings.embed_image_text_pairs(input.doc)  
        res = EmbedDoc1024(text=input.doc, embedding=embed_vector)
    else:
        raise ValueError("Invalid input type")
        

    statistics_dict["opea_service@embedding_multimodal"].append_latency(time.time() - start, None)
    return res


if __name__ == "__main__":
    mm_embedding_endpoint = os.getenv("MM_EMBEDDING_ENDPOINT", "http://localhost:8080")
    embeddings = MMEmbeddings(model=mm_embedding_endpoint)
    print("MM Gaudi Embedding initialized.")
    opea_microservices["opea_service@embedding_multimodal"].start()
