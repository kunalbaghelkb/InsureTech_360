import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.constants import *
from src.exception import CustomException
from src.logger import logging

load_dotenv()

class VectorDBBuilder:
    def __init__(self):
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        self.embed_model_name = os.getenv("GOOGLE_EMBED_MODEL_NAME")

    def create_vector_db(self):
        try:
            logging.info("Starting Vector DB Generation...")

            # Load PDF
            if not os.path.exists(POLICY_PDF_PATH):
                raise FileNotFoundError(f"Policy PDF not found at {POLICY_PDF_PATH}")
            
            logging.info(f"Loading PDF from: {POLICY_PDF_PATH}")
            
            loader = PyPDFLoader(POLICY_PDF_PATH)
            docs = loader.load()
            logging.info(f"Loaded {len(docs)} pages.")

            # Split Text
            logging.info("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            final_documents = text_splitter.split_documents(docs)
            logging.info(f"Created {len(final_documents)} text chunks.")

            # Setup Embeddings
            logging.info("Initializing Embedding Model...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=f"models/{self.embed_model_name}",
                google_api_key=self.google_api_key
            )

            # Create Vectors & Save
            logging.info("Creating FAISS Index ...")
            vectors = FAISS.from_documents(final_documents, embeddings)
            
            # Save Local
            vectors.save_local(FAISS_DB_PATH)
            logging.info(f"Vector DB Saved Successfully at: {FAISS_DB_PATH}")

        except Exception as e:
            raise CustomException(e, sys)