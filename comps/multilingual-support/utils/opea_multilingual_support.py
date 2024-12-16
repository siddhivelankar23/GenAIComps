# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from detector import detect_language
from prompt import set_prompt
from comps import LLMParamsDoc, GeneratedDoc, CustomLogger
import re


logger = CustomLogger("opea_multilingual_support")


class OPEAMultilingualSupport:
    def __init__(self):
        logger.info("Multilingual Support microservice initialized.")

    def run(self, input: GeneratedDoc) -> LLMParamsDoc:
        """
        Detects the language of the query and sets up a translation prompt if needed, without modifying the query.

        Args:
            input (GeneratedDoc): The input document containing the initial query and answer.

        Returns:
            LLMParamsDoc: The generated prompt for translation.
        """
        if not input.prompt.strip():
            logger.error("No initial query provided.")
            raise ValueError("Initial query cannot be empty.")

        if not input.text.strip():
            logger.error("No answer provided from LLM.")
            raise ValueError("Answer from LLM cannot be empty.")


        # Extract question from prompt
        match = re.search(r"### Question:\s*(.*?)\s*(?=### Answer:|$)", input.prompt, re.DOTALL)

        if match:
            extracted_question = match.group(1).strip()  # Remove any leading/trailing whitespace
        else:
            raise ValueError("Question not found in the prompt!")

        # Detect the language of the query
        query_language = detect_language(extracted_question)
        logger.info(f"Detected language of the query: {query_language}")

        # Detect the language of the answer
        answer_language = detect_language(input.text)
        logger.info(f"Detected language of the answer: {answer_language}")

        if query_language == answer_language:
            translation_prompt = set_prompt(query_language, 'en', input.text) # Prevents back-translation to English if RAG LLM generates answer in the same language
        else:
            translation_prompt = set_prompt(query_language, answer_language, input.text)

        # Return the prompt
        return LLMParamsDoc(query=translation_prompt)
