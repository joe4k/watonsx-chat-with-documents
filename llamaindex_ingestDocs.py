import os

from llama_index.core import SimpleDirectoryReader, ServiceContext
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex

from llama_index.core.evaluation import DatasetGenerator

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.memory import ChatMemoryBuffer


# import QueryBundle
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

import chromadb

# Hybrid (vector and keyword) search retriever
# https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

# Given a directory, return a list of files in that directory
# Provide the complete path for each file
# it is assumed all files in this directory should be ingested into 
# knowledge base for RAG
def getDocuments(docDir):

    flist = os.listdir(docDir)
    files = []
    for f in flist:
        fpath = os.path.join(docDir,f)
        if os.path.isfile(fpath):
            files.append(fpath)
        
    return files  

def generate_sample_questions(documents,llm,nqueries):

    # Load Data
    #reader = SimpleDirectoryReader(docdir)
    #documents = reader.load_data()

    nDocs = len(documents)
    # You can choose to generate sample questions on a subset of the documents
    # eval_documents = documents[:10] ==> generates questions using first 10 chunks 
    eval_documents = documents
    data_generator = DatasetGenerator.from_documents(eval_documents,llm=llm)
    eval_questions = data_generator.generate_questions_from_nodes(num = nqueries)

    return eval_questions

# Ingest documents into a vector store and builde a knowledge base index
def ingestDocs_chat(documentDirectory,llm,chunk_size,chunk_overlap,top_k=3,generateQuestions=False):
    docList = getDocuments(documentDirectory)
    print("doc list: ", docList)
    kb_docs = SimpleDirectoryReader(input_files=docList).load_data()
    
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("watsonx2024",get_or_create=True)
    
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    #vector_store.get_nodes()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # define embedding function
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embed_model=embed_model)

    vector_index = VectorStoreIndex.from_documents(kb_docs,storage_context=storage_context,service_context=service_context)

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        
    chat_engine = vector_index.as_chat_engine(chat_mode="condense_plus_context", memory=memory, llm=llm, verbose=True, similarity_top_k=top_k)
    # Do we need to add explicit prompt? like example below?
    # https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/
    #chat_engine = index.as_chat_engine(
    #    chat_mode="condense_plus_context",
    #    memory=memory,
    #    llm=llm,
    #    context_prompt=(
    #        "You are a chatbot, able to have normal interactions, as well as talk"
    #        " about an essay discussing Paul Grahams life."
    #        "Here are the relevant documents for the context:\n"
    #        "{context_str}"
    #        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    #    ),
    #    verbose=False,
    #)

    if generateQuestions == True:
        sample_questions = generate_sample_questions(kb_docs,llm,10)
    else:
        sample_questions = []
        
    return chat_engine, sample_questions


# Ingest documents into a vector store and builde a knowledge base index
def ingestDocs_hybrid(documentDirectory,llm,chunk_size,chunk_overlap,top_k=3):
    docList = getDocuments(documentDirectory)
    print("doc list: ", docList)
    kb_docs = SimpleDirectoryReader(input_files=docList).load_data()
    
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("watsonx2024",get_or_create=True)
    
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    #vector_store.get_nodes()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # define embedding function
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embed_model=embed_model)
    

    

    vector_index = VectorStoreIndex.from_documents(kb_docs,storage_context=storage_context,service_context=service_context)
    keyword_index = SimpleKeywordTableIndex.from_documents(kb_docs,storage_context=storage_context,service_context=service_context)
    
    # define custom retriever
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

    # define response synthesizer
    response_synthesizer = get_response_synthesizer(llm=llm)

    # assemble query engine
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    # vector query engine
    vector_query_engine = RetrieverQueryEngine(
        retriever=vector_retriever,
        response_synthesizer=response_synthesizer,
    )
    # keyword query engine
    keyword_query_engine = RetrieverQueryEngine(
        retriever=keyword_retriever,
        response_synthesizer=response_synthesizer,
    )


    #kb_index = VectorStoreIndex.from_documents(
    #    kb_docs, storage_context=storage_context, service_context=service_context
    #)

    #kb_engine = kb_index.as_query_engine()

    return custom_query_engine, vector_query_engine, keyword_query_engine