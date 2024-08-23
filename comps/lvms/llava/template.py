# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class ChatTemplate:
    @staticmethod
    def generate_simple_prompt(prompt):
        return f"<image>\nUSER: {prompt}\nASSISTANT:"
    
    @staticmethod
    def generate_rag_prompt(context, question):
        template = """USER: <image>
The transcript associated with the image is '{context}'. {question} 
ASSISTANT:"""
        return template.format(context=context, question=question)