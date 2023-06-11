from langchain.document_loaders import OnlinePDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_data(path:str,extension:str ,pdf_url) :
    if path.startswith('http'):
        loader = OnlinePDFLoader(pdf_url).load()
        return loader
    else:
        loader = DirectoryLoader('<path to pdf>', glob=f'**/*.{extension}').load()
        return loader

def text_split(chunk_size, chunk_overlap, pdf_docs):
    # *Split text into chunks *
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create OpenAIEmbeddings and FAISS objects. Vectorize the chunks created above and save.
    documents = text_splitter.split_documents(pdf_docs)
    return documents



pdf_url ='<what is the url/path>'
loader = ingest_data(pdf_url,'pdf')
pdf_docs = loader
# Create OpenAIEmbeddings. Vectorize the chunks created above and save.
documents = text_split(1000, 200, pdf_docs)
print ('splits docs', documents)