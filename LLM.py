from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from Global_Variables import host_url, api_key, project_id

class LLM:
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 250,
        GenParams.TEMPERATURE: 0.5
    }
    
    credentials = {
        "url": host_url,
        "apikey": api_key
    }

    project_id = project_id

    def __init__(self, model_id):
        self.model_id = model_id
        
        self.model = Model(
            model_id=self.model_id,
            credentials=self.credentials,
            params=self.parameters,
            project_id=self.project_id
        )
        
        self.llama_3_llm = WatsonxLLM(model=self.model)
        
        self.qa = None
    
    def constructQA(self, retriever, memory):
        # self.qa = RetrievalQA.from_chain_type(llm=self.llama_3_llm,
        #                                       chain_type='stuff',
        #                                       retriever=retriever,
        #                                       return_source_documents=False)
        self.qa = ConversationalRetrievalChain.from_llm(llm=self.llama_3_llm,
                                                        chain_type='stuff',
                                                        retriever=retriever,
                                                        memory=memory,
                                                        get_chat_history=lambda h : h,
                                                        return_source_documents=False)
    
    def getResponse(self, query, history):
        result = self.qa.invoke({"question": query, "chat_history": history})
        return result