# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from dotenv import load_dotenv
from fastapi import HTTPException

from comps import (
    CustomLogger,
    LLMParamsDoc,
    GeneratedDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from utils.opea_multilingual_support import OPEAMultilingualSupport

logger = CustomLogger("multilingual_support")
logflag = os.getenv("LOGFLAG", False)


# Initialize an instance of the language detector class with environment variables.
opea_language_detector = OPEAMultilingualSupport()

# Register the microservice with the specified configuration.
@register_microservice(
    name="opea_service@multilingual_support",
    service_type=ServiceType.MULTILINGUAL_SUPPORT,
    endpoint="/v1/multilingual_support",
    host='0.0.0.0',
    port=int(os.getenv('LANGUAGE_DETECTION_USVC_PORT', default=8001)),
    input_datatype=GeneratedDoc,
    output_datatype=LLMParamsDoc,
)
@register_statistics(names=["opea_service@multilingual_support"])

async def process(input: GeneratedDoc) -> LLMParamsDoc:
    """
    Process the input document using the OPEALanguageDetector.

    Args:
        input (GeneratedDoc): The input document to be processed.

    Returns:
        LLMParamsDoc: The processed document with LLM parameters.
    """
    start = time.time()
    try:
        # Pass the input to the 'run' method of the microservice instance
        res = opea_language_detector.run(input)
    except ValueError as e:
        logger.exception(f"An internal error occurred while processing: {str(e)}")
        raise HTTPException(status_code=400,
                            detail=f"An internal error occurred while processing: {str(e)}"
        )
    except Exception as e:
         logger.exception(f"An error occurred while processing: {str(e)}")
         raise HTTPException(status_code=500,
                             detail=f"An error occurred while processing: {str(e)}"
    )
    statistics_dict["opea_service@multilingual_support"].append_latency(time.time() - start, None)
    return res


if __name__ == "__main__":
    # Start the microservice
    tei_reranking_endpoint = os.getenv("MULTILINGUAL_SUPPORT_ENDPOINT", "http://localhost:8081")
    opea_microservices["opea_service@multilingual_support"].start()
    logger.info(f"Started OPEA Microservice: {"opea_service@multilingual_support"}")
