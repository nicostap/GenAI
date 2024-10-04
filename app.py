import streamlit as st
# import request
from ta import (trend, momentum, volatility, volume)
import yfinance as yf
import pandas as pd
from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex, Settings, Document, load_index_from_storage)
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
from llama_index.core import PromptTemplate
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from data_utils import CustomTools

class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="jinaai/jina-embeddings-v2-base-code", vector_store=None, cTools=CustomTools()):
        self.Settings = self.set_setting(llm, embedding_model)

        self.memory = self.create_memory()

        self.index = self.load_data('./docs')
        self.cTools = cTools

        # self.chat_engine = self.create_chat_engine(self.index)
        self.chat_engine = self.create_react_agent(self.index)

    def set_setting( _arg, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = FastEmbedEmbedding(
            model_name=embedding_model, cache_dir="./fastembed_cache")
        Settings.system_prompt = """
       You are a multi-lingual expert system named Jarvis. You have extensive knowledge in investment analysis, specializing in both 
       technical and fundamental analysis across various asset classes, including stocks, cryptocurrencies, forex, 
       and more. Your primary task is to deliver insightful, data-driven analysis and tailored recommendations based 
       on user queries regarding investments. You can respond fluently in both English and Indonesian. 
       Your responsibilities include:
    
        1. **Technical Analysis:**
           - Provide insights based on technical indicators such as RSI, MACD, Bollinger Bands, Moving Averages, etc.
           - Analyze chart patterns and trends to generate buy/sell signals.
           - Explain the significance of each technical indicator and how it informs trading decisions.
    
        2. **Fundamental Analysis:**
           - Offer detailed analyses of financial statements, including P/E ratios, EPS, revenue growth, and other key metrics.
           - Evaluate the intrinsic value of stocks or cryptocurrencies based on fundamental data.
           - Explain how fundamental factors influence market movements and investment decisions.
    
        3. **Educational Support:**
           - Educate users on the basics of trading, including key concepts, strategies, and best practices.
           - Provide tutorials on how to interpret technical and fundamental indicators.
           - Answer beginner questions with clear and concise explanations to foster learning.
    
        4. **Market Predictions:**
           - Utilize historical and real-time data to offer market outlooks and trend predictions.
           - Clearly communicate the speculative nature of predictions and encourage users to conduct their own research.
           - Explain the factors contributing to predicted market movements.
    
        5. **User Interaction:**
           - Engage in multi-turn conversations, maintaining context to provide coherent and relevant responses.
           - Adapt explanations based on the user's level of expertise, offering advanced insights to experienced traders and simplified explanations to beginners.
    
            Ensure all information is accurate, up-to-date, and presented in an unbiased manner. Prioritize user understanding and empower them to make informed trading decisions.
        """

        return Settings


    def create_react_system_prompt(self, prompt=None):
        if prompt!=None:
            react_system_header_str = prompt

        else:         
            react_system_header_str = """\
            
            ## Tools
            You have access to a wide variety of tools. You are responsible for using
            the tools in any sequence you deem appropriate to complete the task at hand.
            This may require breaking the task into subtasks and using different tools
            to complete each subtask.
            
            You have access to the following tools:
            {tool_desc}
            
            ## Output Format
            To answer the question, please use the following format.
            
            ```
            Thought: I need to use a tool to help me answer the question.
            Action: tool name (one of {tool_names}) if using a tool.
            Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
            ```
            
            Please ALWAYS start with a Thought.
            
            Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.
            
            If this format is used, the user will respond in the following format:
            
            ```
            Observation: tool response
            
            ```
            You should keep repeating the above format until you have enough information
            to answer the question without using any more tools. At that point, you MUST respond
            in the one of the following two formats:
            
            ```
            Thought: I can answer without using any more tools.
            Answer: [your answer here]
            ```
            
            ```
            Thought: I cannot answer the question with the provided tools.
            Answer: Sorry, I cannot answer your query.
            ```
            
            ## Additional Rules
            - You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.
            
            ## Current Conversation
            Below is the current conversation consisting of interleaving human and assistant messages.
            
            """
        react_system_prompt = PromptTemplate(react_system_header_str)
        return react_system_prompt

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )


    def create_react_agent(self, index):
        get_stock_data_tool = FunctionTool.from_defaults(fn=self.cTools.get_stock_data)
        analyze_stock_tool = FunctionTool.from_defaults(fn=self.cTools.analyze_stock)
        analyze_fundamental_tool = FunctionTool.from_defaults(fn=self.cTools.analyze_fundamental)
        
        tools = [get_stock_data_tool, analyze_stock_tool, analyze_fundamental_tool]
        agent = ReActAgent.from_tools(
            tools,
            chat_mode="react",
            verbose=True,
            memory=self.memory,
            react_system_prompt=self.create_react_system_prompt(),
            # retriever=index.as_retriever(),
            llm=Settings.llm)
        return agent

    def split_text_into_chunks(self, text, chunk_size=2000):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def create_index_from_text(self, text_chunks):
        # Create documents and add them to the index
        documents = [Document(text=chunk) for chunk in text_chunks]
        parser = SimpleNodeParser.from_defaults(chunk_size=200)
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)

        return index

    @st.cache_resource(show_spinner=False)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
            # Read & load document from folder
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            documents = reader.load_data()

        if vector_store is None:
            index = VectorStoreIndex.from_documents(documents)

        else:
            index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
        return index

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}


st.title("Market Analysis Assistance")

chatbot = Chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello there 👋!\n\n I'm Jarvis, your intelligent financial assistant. I can help you perform technical and fundamental analysis on various investment products, educate you on trading basics, and even provide market predictions. How can I assist you today?"}
    ]

print(chatbot.chat_store.store)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chatbot.set_chat_history(st.session_state.messages)

if prompt := st.chat_input("How can i help you?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = ""
        response = chatbot.chat_engine.chat(prompt)
        st.markdown(response.response)
        # response.print_response_stream()

    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})

