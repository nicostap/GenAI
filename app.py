import streamlit as st
from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core import StorageContext
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
import pdfplumber
import os
import fitz

class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="jinaai/jina-embeddings-v2-base-code", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./persist"),
            vector_store=SimpleVectorStore.from_persist_dir(
                persist_dir="./persist"
            ),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="./persist"),
        )
        self.index = load_index_from_storage(storage_context)

        self.memory = self.create_memory()

        # self.index = self.load_data('./docs')
        # self.index.storage_context.persist(persist_dir="./persist")

        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(_arg, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = FastEmbedEmbedding(
            model_name=embedding_model, cache_dir="./fastembed_cache")
        Settings.system_prompt = """
                                You are a multi-lingual expert system named Roni who has various knowledge on love relationship.
                                Your task is to only help answer any concern that the user have regarding love relationship or something similar.
                                """
        return Settings

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        return text

    def split_text_into_chunks(self, text, chunk_size=2000):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def create_index_from_text(self, text_chunks):
        # Create documents and add them to the index
        documents = [Document(text=chunk) for chunk in text_chunks]
        parser = SimpleNodeParser.from_defaults(chunk_size=200)
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)

        return index

    def process_pdfs_in_directory(self, directory_path):
        """Process all PDFs in the specified directory and create an index."""
        all_text_chunks = []

        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory_path, filename)
                print(f"Processing {filename}...")

                # Extract text from PDF
                text = self.extract_text_from_pdf(pdf_path)

                # Split the text into chunks
                text_chunks = self.split_text_into_chunks(text)

                # Add text chunks to the list
                all_text_chunks.extend(text_chunks)

        # Create the index from all text chunks
        index = self.create_index_from_text(all_text_chunks)
        return index

    def load_data(self, directory_path):
        with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
            index = self.process_pdfs_in_directory(directory_path)
            return index

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}


st.title("Relationship Consultation with Roni")


chatbot = Chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello there 👋!\n\n Good to see you, my name is Roni. Feel free to ask me anything regarding relationship 😁"}
    ]

print(chatbot.chat_store.store)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chatbot.set_chat_history(st.session_state.messages)

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = chatbot.chat_engine.chat(prompt)
        st.markdown(response.response)

    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response.response})