def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

import wget

import Global_Variables
from Preprocessing import Preprocessing
from LLM import LLM

def load_documents(file_name, url):
    file_name = file_name
    url = url
    wget.download(url, out=file_name)
    print('file downloaded')
    
history = []
    
def qa(preprocessing):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = LLM(Global_Variables.model_id)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    llm.constructQA(preprocessing.docsearch.as_retriever(), memory)
    
    while True:
        query = input("Question: ")
        if (query.lower() in ["quit", "exit", "bye"]):
            print("Answer: Goodbye!")
            break
        result = llm.getResponse(query, history)
        history.append((query, result['answer']))
        print("Answer: ", result['answer'])
    
def main():
    preprocessing = Preprocessing()
    preprocessing.split(Global_Variables.file_name)
    preprocessing.embed_and_store()
    
    qa(preprocessing)
        
if __name__ == '__main__':
    main()