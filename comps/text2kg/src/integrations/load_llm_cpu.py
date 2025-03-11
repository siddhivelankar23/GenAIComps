import os
import yaml
import shutil
from pyprojroot import here
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class load_llm():
    def __init__(
            self,
            llm_model_engine: str,
            embedding_model_engine: str,
    ) -> None:
        self.embedding_model_engine = "BAAI/bge-small-en-v1.5"
        self.llm_engine = "HuggingFaceH4/zephyr-7b-alpha"
     
    def messages_to_prompt(messages):
       prompt = ""
       for message in messages:
         if message.role == 'system':
           prompt += f"<|system|>\n{message.content}</s>\n"
         elif message.role == 'user':
           prompt += f"<|user|>\n{message.content}</s>\n"
         elif message.role == 'assistant':
           prompt += f"<|assistant|>\n{message.content}</s>\n"
     
       # ensure we start with a system prompt, insert blank if needed
       if not prompt.startswith("<|system|>\n"):
         prompt = "<|system|>\n</s>\n" + prompt
     
       # add final assistant prompt
       prompt = prompt + "<|assistant|>\n"
       return prompt

    def load_llm_models(self):
          model_id2 = "HuggingFaceH4/zephyr-7b-alpha"
          llm = HuggingFaceLLM(
            model_name=model_id2,
            tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
            query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
            context_window=3900,
            max_new_tokens=256,
            # model_kwargs={"quantization_config": quantization_config},
            # tokenizer_kwargs={},
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            messages_to_prompt=self.messages_to_prompt,
            device_map="auto",
          )
     
          Settings.llm = llm
          Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
          return llm

    def load_embed_model(self):
          embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
          return embed_model

#################################################################################################
##
##    def llm_pipe(self):
##        print(f'running llm infernece model load')
##        model_id1 = "HuggingFaceH4/zephyr-7b-alpha"
##        tokenizer = AutoTokenizer.from_pretrained(model_id1)
##        model = AutoModelForCausalLM.from_pretrained(model_id1)
##        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
##        hfpipe = HuggingFacePipeline(pipeline=pipe)
##        print(f'running llm infernece set up done ')
##        return hfpipe
##
##    def load_all_models(self):
##       llm = self.load_llm_models()
##       embed_model = self.load_embed_model()
##       hfpipe = self.llm_pipe()
##       result = hfpipe('who is god?')
##       print(result)
##       print(f'running llm infernece set up done ')
##       return llm, embed_model, hfpipe
##       
