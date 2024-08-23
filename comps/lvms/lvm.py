# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import json
import os
import time

import requests
from typing import Union

from comps import (
    CustomLogger,
    LVMDoc,
    SearchedMultimodalDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("lvm")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@lvm",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9399,
    output_datatype=TextDoc,
)
@register_statistics(names=["opea_service@lvm"])
async def lvm(request: Union[LVMDoc, SearchedMultimodalDoc]):
    if logflag:
        logger.info(request)
    start = time.time()

    if isinstance(request, SearchedMultimodalDoc):
        retrieved_metadata = request.metadata[0]

        img_b64_str = retrieved_metadata["b64_img_str"]
        prompt = request.initial_query
        context = retrieved_metadata["transcript_for_inference"]
        max_new_tokens = 512

        inputs = {"img_b64_str": img_b64_str, "prompt": prompt, "context": context, "max_new_tokens": max_new_tokens}
    else:
        img_b64_str = request.image
        prompt = request.prompt
        max_new_tokens = request.max_new_tokens

        inputs = {"img_b64_str": img_b64_str, "prompt": prompt, "max_new_tokens": max_new_tokens}

    # forward to the LLaVA server
    response = requests.post(url=f"{lvm_endpoint}/generate", data=json.dumps(inputs), proxies={"http": None})

    statistics_dict["opea_service@lvm"].append_latency(time.time() - start, None)
    result = response.json()["text"]
    if logflag:
        logger.info(result)
    return TextDoc(text=result)


if __name__ == "__main__":
    lvm_endpoint = os.getenv("LVM_ENDPOINT", "http://localhost:8399")

    logger.info("[LVM] LVM initialized.")
    opea_microservices["opea_service@lvm"].start()
