from langchain_neo4j import Neo4jGraph
##import nest_asyncio
##nest_asyncio.apply()


class PrepareGraphDB:
    """
    A class for preparing and saving a GraphDB 
    """


    def __init__(
            self,
            llm,
            embed_model,
            data_directory: str,
            persist_directory: str,
    ) -> None:
        """
        Initialize the PrepareGraphDB instance.

        Parameters:
            data_directory (str or List[str]): The directory or list of directories containing the documents.
            persist_directory (str): The directory to save the VectorDB.
            embedding_model_engine (str): The engine for OpenAI embeddings.
            chunk_size (int): The size of the chunks for document processing.
            chunk_overlap (int): The overlap between chunks.

        """
        self.llm = llm
        self.embed_model = embed_model
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        print(f' loading and preparing llm and embedding models')

#-------------------------------------------------------------------------------
#   Link up to Neo4j
#-------------------------------------------------------------------------------
    def __neo4j_link(self,NEO4J_URL, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE):
          import os
          from langchain_community.graphs import Neo4jGraph
     
          os.environ["NEO4J_URL"] = NEO4J_URL
          os.environ["NEO4J_URI"] = NEO4J_URI
          os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
          os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
          os.environ["NEO4J_DATABASE"] = NEO4J_DATABASE


          graph_store = Neo4jGraph(
              username=NEO4J_USERNAME,
              password=NEO4J_PASSWORD,
              url=NEO4J_URL,
          )
     
          return graph_store

          #########################################################################
          #       Cleanup and start from scratch
          #########################################################################
          # Delete everything in a database
          cypher = """
          MATCH (n) DETACH DELETE n
          """
          graph_store.query(cypher)
          
          print("## Existing graph_store schema...")
          print(graph_store.schema)
          
          print("Deleting all nodes...")
          # Match all nodes in the graph_store
          cypher = """
            MATCH (n)
            RETURN count (n)
            """
          result = graph_store.query(cypher)
          
          print("Dropping all constraints...")
          for constraint in graph_store.query('SHOW CONSTRAINTS'):
              graph_store.query(f"DROP CONSTRAINT {constraint['name']}")
          
          print("Dropping all indexes...")
          for index in graph_store.query('SHOW INDEXES'):
              print(f"Removing index {index['name']}:")
              graph_store.query(f"""
                  DROP INDEX `{index['name']}`
              """)
          
          print()
          print("## Blank schema...")
          graph_store.refresh_schema()
          print(graph_store.schema)

    def prepare_insert_graphdb(self):
         NEO4J_URL = "neo4j://localhost:7687"
         NEO4J_URI = "neo4j://localhost:7687"
         NEO4J_USERNAME = "neo4j"
         NEO4J_PASSWORD = "intel123"
         NEO4J_DATABASE = "neo4j"
         graph_store = self.__neo4j_link(NEO4J_URL, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

         cypher_cleanup = """
          MATCH (n) DETACH DELETE n
         """
         cypher_insert = """
          LOAD CSV WITH HEADERS FROM  'file:///opt/neo4j/import/movie.csv'   // Load CSV data from a file specified by $movie_directory
          AS row                                                      // Each row in the CSV will be represented as 'row'

          MERGE (m:Movie {id:row.movieId})                            // Merge a Movie node with the id from the row
          SET m.released = date(row.released),                        // Set the 'released' property of the Movie node to the date from the row
              m.title = row.title,                                    // Set the 'title' property of the Movie node to the title from the row
              m.tagline = row.tagline,                                // Set the 'tagline' property of the Movie node to the tagline from the row
              m.imdbRating = toFloat(row.imdbRating)                  // Convert the 'imdbRating' from string to float and set it as the property
          
          FOREACH (director in split(row.director, '|') |             // For each director in the list of directors from the row (split by '|')
              MERGE (p:Person {name:trim(director)})                  // Merge a Person node with the director's name from the row, trimming any extra spaces
              MERGE (p)-[:DIRECTED]->(m))                             // Create a DIRECTED relationship from the director to the Movie
          
          FOREACH (actor in split(row.actors, '|') |                  // For each actor in the list of actors from the row (split by '|')
              MERGE (p:Person {name:trim(actor)})                     // Merge a Person node with the actor's name from the row, trimming any extra spaces
              MERGE (p)-[:ACTED_IN]->(m))                             // Create an ACTED_IN relationship from the actor to the Movie
          
          FOREACH (genre in split(row.genres, '|') |                  // For each genre in the list of genres from the row (split by '|')
              MERGE (g:Genre {name:trim(genre)})                      // Merge a Genre node with the genre's name from the row, trimming any extra spaces
              MERGE (m)-[:IN_GENRE]->(g))                             // Create an IN_GENRE relationship from the Movie to the Genre
          
          MERGE (l:Location {name:trim(row.location)})
          MERGE (m)-[:WAS_TAKEN_IN]->(l)
          
          MERGE (s:SimilarMovie {name:trim(row.similar_movie)})
          MERGE (m)-[:IS_SIMILAR_TO]->(s)
         """
         print(f'Cleaning up graph db')
         graph_store.query(cypher_cleanup)
         print(f'Done cleaning up graph db')

         graph_store.query(cypher_insert)
         print(f'Completed reading document and inserting into graphdb')
         print(f'The following is the graph schema \n\n {graph_store.schema}')
         print("Preparing graphdb...")
         print("GraphDB is created and saved.")
         return graph_store

if __name__ == "__main__":
     gdb = PrepareGraphDB(
            llm = "HuggingFaceH4/zephyr-7b-alpha",
            embed_model= "BAAI/bge-small-en-v1.5",
            data_directory = "data/",
            persist_directory = "data/vectordb"
            )
     graph_store = gdb.prepare_insert_graphdb()
     question = "MATCH (:Movie {title: 'Casino'})<-[:ACTED_IN]-(actor:Person) RETURN actor.name AS actor"
     result = graph_store.query(question)
     print(result)
     print('I am done')
