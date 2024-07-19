# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Optional, Tuple, Union

#from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from embeddings.BridgeTowerEmbeddings import BridgeTowerEmbeddings, MMEmbeddings
from langchain_community.vectorstores import Redis
from langsmith import traceable
from redis_config import EMBED_MODEL, INDEX_NAME, REDIS_URL

from comps import (
    EmbedDoc1024,
    SearchedMultimodalDoc,
    ServiceType,
    TextDoc,
    ImageDoc, 
    TextImageDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

mm_embedding_endpoint = os.getenv("MM_EMBEDDING_ENDPOINT")


@register_microservice(
    name="opea_service@mm_retriever_redis",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@traceable(run_type="retriever")
@register_statistics(names=["opea_service@mm_retriever_redis"])
def retrieve(input: EmbedDoc1024) -> SearchedMultimodalDoc:
    start = time.time()
    # check if the Redis index has data
    if vector_db.client.keys() == []:
        result = SearchedMultimodalDoc(retrieved_docs=[], initial_query=input.text)
        statistics_dict["opea_service@mm_retriever_redis"].append_latency(time.time() - start, None)
        return result

    # if the Redis index has data, perform the search
    if input.search_type == "similarity":
        search_res = vector_db.similarity_search_by_vector(embedding=input.embedding, k=input.k)
    elif input.search_type == "similarity_distance_threshold":
        if input.distance_threshold is None:
            raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
        search_res = vector_db.similarity_search_by_vector(
            embedding=input.embedding, k=input.k, distance_threshold=input.distance_threshold
        )
    elif input.search_type == "similarity_score_threshold":
        docs_and_similarities = vector_db.similarity_search_with_relevance_scores(
            query=input.text, k=input.k, score_threshold=input.score_threshold
        )
        search_res = [doc for doc, _ in docs_and_similarities]
    elif input.search_type == "mmr":
        search_res = vector_db.max_marginal_relevance_search(
            query=input.text, k=input.k, fetch_k=input.fetch_k, lambda_mult=input.lambda_mult
        )
    searched_docs = []
    for r in search_res:
        if isinstance(r, TextDoc):
            searched_docs.append(TextDoc(text=r.page_content))
        elif isinstance(r, ImageDoc):  
            searched_docs.append(ImageDoc(image=r.image_data))
        elif isinstance(r, TextImageDoc):  
            searched_docs.append(TextImageDoc(Tuple[Union[r.page_content, r.image_data]]))

    result = SearchedMultimodalDoc(retrieved_docs=searched_docs, initial_query=input.text)
    statistics_dict["opea_service@mm_retriever_redis"].append_latency(time.time() - start, None)
    return result


if __name__ == "__main__":
    # Create vectorstore
    if mm_embedding_endpoint:
        # create embeddings using MM endpoint service
        embeddings = MMEmbeddings(model=mm_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embeddings = MMEmbeddings(model_name=EMBED_MODEL)

    vector_db = Redis(embedding=embeddings, index_name=INDEX_NAME, redis_url=REDIS_URL)
    opea_microservices["opea_service@mm_retriever_redis"].start()
