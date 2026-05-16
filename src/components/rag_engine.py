import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.constants import FAISS_DB_PATH
from src.exception import CustomException
from src.logger import logging

load_dotenv()

@dataclass
class RAGConfig:
    vector_db_path = FAISS_DB_PATH
    google_api_key = os.getenv("GEMINI_API_KEY")
    google_llm_model_name = os.getenv("GOOGLE_LLM_MODEL_NAME")
    google_embed_model_name = os.getenv("GOOGLE_EMBED_MODEL_NAME")
    
class RAGEngine:
    def __init__(self):
        self.config = RAGConfig()
        self.rag_chain = None
        
        self._initialize_rag_system()
        
    def _initialize_rag_system(self):
        '''
        Loads the Vector DB and sets up the LCEL (Runnable) Chain.
        '''
        
        try:
            logging.info("Initializing RAG Engine...")
            
            if not self.config.google_api_key:
                raise Exception("GEMINI_API_KEY not found in .env file")
            
            # ---> Setup Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model=f"models/{self.config.google_embed_model_name}", 
                google_api_key=self.config.google_api_key
            )
            
            # ---> Load Vector DB
            logging.info(f"Loading Vector DB from {self.config.vector_db_path}")
            vector_store = FAISS.load_local(
                self.config.vector_db_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Convert to Retriever (Optimized: Retrieve only top 3 chunks for speed)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            
            # ---> Setup LLM (Optimized for performance)
            llm = ChatGoogleGenerativeAI(
                model=f"{self.config.google_llm_model_name}", 
                temperature=0.1, # Lower temperature for faster, more direct responses
                google_api_key=self.config.google_api_key,
                convert_system_message_to_human=True
            )
            
            # ---> Prompt Template (100% Optimized for Accuracy and Formatting)
            prompt = ChatPromptTemplate.from_template("""
            SYSTEM: You are a professional Insurance Claims Expert for 'InsureTech 360'. 
            Your goal is to provide accurate, concise, and clear information based ONLY on the provided policy documents.

            RULES:
            1. Use the 'CONTEXT' provided below to answer the 'USER QUESTION'.
            2. If the information is not present in the CONTEXT, state: "I'm sorry, but I cannot find that information in the current policy document."
            3. Do NOT use outside knowledge or hallucinate details.
            4. Use bullet points if the answer contains multiple steps or items for better readability.

            ---
            CONTEXT:
            {context}
            ---
            USER QUESTION: 
            {question}

            ASSISTANT RESPONSE:
            """)
            
            # ---> Build the Runnable Chain
            # Logic: Retriever -> Prompt -> LLM -> String Output
            self.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            logging.info("RAG Engine Initialized Successfully!")
            
        except Exception as e:
            logging.error(f"Error initializing RAG: {e}")
            raise CustomException(e, sys)
        
    def get_response(self, query):
        """
        Takes user query and returns AI answer.
        """
        try:
            if not self.rag_chain:
                return "System Error: RAG Chain not initialized."
            
            logging.info(f"Processing Query: {query}")
            
            # New Style Invocation
            response = self.rag_chain.invoke(query)
            
            return response
        
        except Exception as e:
            raise CustomException(e, sys)