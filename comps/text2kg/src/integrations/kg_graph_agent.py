import neo4j, os
from typing import Literal
from llama_index.core import KnowledgeGraphIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore, Neo4jGraphStore
from llama_index.core.prompts import PromptTemplate
from comps.text2kg.src.integrations.load_llm_cpu import load_llm
import subprocess
import nest_asyncio
nest_asyncio.apply()

class GenerateKG:
    def __init__(self, llm, embedding_model , data_directory):
         self.data_directory    = data_directory
         models                 = load_llm(llm_model_engine=llm, embedding_model_engine=embedding_model)
         self.llm               = models.load_llm_models()
         self.embed_model       = models.load_embed_model()
         Settings.llm           = self.llm
         Settings.embed_model   = self.embed_model
         print(f' loading and preparing llm and embedding models')

    def __load_docs(self):
        TEMP_DIR = os.path.join(os.getcwd(), "data")
        FILE_URL = "https://gist.githubusercontent.com/wey-gu/75d49362d011a0f0354d39e396404ba2/raw/0844351171751ebb1ce54ea62232bf5e59445bb7/paul_graham_essay.txt"
        command = ["wget", "-P", TEMP_DIR, FILE_URL]
        try:
             result = subprocess.run(command, check=True, capture_output=True, text=True)
             print(f"Download successful. Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
             print(f"Download failed. Error:\n{e.stderr}")


        #text = open(f"{TEMP_DIR}/paul_graham_essay.txt").read()
        #encoded_data2 = quote(text)
        #reader = SimpleDirectoryReader(input_dir=self.data_directory)
        reader = SimpleDirectoryReader(input_dir=TEMP_DIR)
        documents = reader.load_data()
        print(f'loading documents')
        #print(f'READING DOCS {documents[:1000]}')
        return documents
#-------------------------------------------------------------------------------
#   Link up to Neo4j
#-------------------------------------------------------------------------------
    def __neo4j_link(self,NEO4J_URL, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE):
          import os
          import neo4j
          from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
     
          os.environ["NEO4J_URL"] = NEO4J_URL
          os.environ["NEO4J_URI"] = NEO4J_URI
          os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
          os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
          os.environ["NEO4J_DATABASE"] = NEO4J_DATABASE
    
          graph_store = Neo4jGraphStore(
                  username=NEO4J_USERNAME,
                  password=NEO4J_PASSWORD,
                  url=NEO4J_URL,
                  database=NEO4J_DATABASE,
          )
          return graph_store
     
    def __graph_index(self, documents, llm,embed_model,graph_store):
          # best practice to use upper-case
          entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
          relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]
     
          # define which entities can have which relations
          validation_schema = {
              "PERSON": ["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"],
              "PLACE": ["HAS", "PART_OF", "WORKED_AT"],
              "ORGANIZATION": ["HAS", "PART_OF", "WORKED_WITH"],
          }
    
          storage_context = StorageContext.from_defaults(graph_store=graph_store)
          neo4j_index = KnowledgeGraphIndex.from_documents(
                  documents=documents,
                  max_triplets_per_chunk=3,
                  storage_context=storage_context,
                  embed_model=embed_model,
                  include_embeddings=True
          )
          return neo4j_index
     
    def __create_index(self,documents,embed_model,llm):
        NEO4J_URL = "neo4j://localhost:7687"
        NEO4J_URI = "neo4j://localhost:7687"
        NEO4J_USERNAME = "neo4j"
        NEO4J_PASSWORD = "intel123"
        NEO4J_DATABASE = "neo4j"
        graph_store = self.__neo4j_link(NEO4J_URL, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
        neo4j_index = self.__graph_index(documents, llm, embed_model, graph_store)
        print(f" neo4j index {neo4j_index.index_struct}")
        print(f'creating graph index for documents')
        return neo4j_index

    def prepare_and_save_graphdb(self):
        """
        Load, chunk, and create graph and load it into neo4j database
        """
        print(f'entering prepare and save for structured data')
        docs = self.__load_docs()
        neo4j_index = self.__create_index(docs,self.embed_model,self.llm)
        print("Preparing graphdb...")
        print("GraphDB is created and saved.")
        return neo4j_index

