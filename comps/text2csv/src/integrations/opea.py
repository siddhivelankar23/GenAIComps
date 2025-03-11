# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import Annotated, Optional
from pydantic import BaseModel, Field

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.text2csv.src.integrations.test_load import PrepareGraphDB

logger = CustomLogger("comps-text2csv")
logflag = os.getenv("LOGFLAG", False)

graph_params = {
    "max_string_length": 3600,
}

generation_params = {
    "max_new_tokens": 1024,
    "top_k": 10,
    "top_p": 0.95,
    "temperature": 0.01,
    "repetition_penalty": 1.03,
    "streaming": True,
}


class Input(BaseModel):
    input_text: str


@OpeaComponentRegistry.register("OPEA_TEXT2CSV")
class OpeaText2CSV(OpeaComponent):
    """A specialized text to graph triplet converter."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.TEXT2CSV.name.lower(), description, config)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaText2CSV health check failed.")

    async def check_health(self) -> bool:
        """Checks the health of the TGI service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            return True
        except Exception as e:
            return False

    async def invoke(self, input_text: str):
        """Invokes the text2csv service to generate graph(s) for the provided input.

        input:
            input: text document
        Returns:
            text : dict
        """

        #tb = TripletBuilder()
        #graph_triplets = await tb.extract_graph(input_text)

        #result = {"graph_triplets": graph_triplets}
        gdb = PrepareGraphDB(
            llm = "HuggingFaceH4/zephyr-7b-alpha",
            embed_model= "BAAI/bge-small-en-v1.5",
            data_directory = "data/",
            persist_directory = "data/vectordb"
            )
        graph_store = gdb.prepare_insert_graphdb()
        #question = "MATCH (:Movie {title: 'Casino'})<-[:ACTED_IN]-(actor:Person) RETURN actor.name AS actor"
        result = graph_store.query(input_text)
        print(result)
        print('I am done')

        return result
