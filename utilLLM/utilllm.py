import os
from typing import Any, Dict, List
import pinecone
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.document_loaders import OnlinePDFLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone


# LOAD index from Pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

index_pinecone = "example-index-smaller-chunk"

def llm_doc_search(query: str, chat_history: List[Dict[str, Any]] = []):
    # Load the LangChain.
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    # docsearch = Pinecone.from_existing_index(index_pinecone ,embeddings,text_key='text' )
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        # index_name=os.environ["PINECONE_INDEX_NAME"],
        index_name=index_pinecone
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})
