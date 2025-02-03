from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from Global_Variables import file_name

class Preprocessing:
    def __init__(self):
        loader = None
        documents = None
        text_splitter = None
        texts = None
        embeddings = None
        docsearch = None
    
    def split(self, file_name):
        self.loader = TextLoader(file_name)
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = self.text_splitter.split_documents(self.documents)
        print(len(self.texts))

    def embed_and_store(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.docsearch = Chroma.from_documents(self.texts, self.embeddings)
        print('document ingested')