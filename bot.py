from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

import sys
from pydantic import BaseModel, Field
from typing import List
from llama_index.program.lmformatenforcer import (
    LMFormatEnforcerPydanticProgram,
)
from llama_index.llms.llama_cpp import LlamaCPP


class Query(BaseModel):
    input: str

class InputBot:
    def __init__(self):
        self.llm = LlamaCPP()

        self.agent = LMFormatEnforcerPydanticProgram(
            output_cls=Query,
            prompt_template_str=(
                """
                Your response should be according to the following json schema: \n
                {json_schema}\n"

                # Instruction
                1. Receive the user prompt using the user's prompt {prompt} as inspiration.
                1. Identify the language of the conversation from user's prompt : Determine the language the user is speaking.
                2. Understand the user's prompt: Summarize what the user wants to know or ask.
                3. Generate an example Query with an "input" property that follows the following strict format :
                    "Give me a list of publications about [topic]."
                    Ensure that this query is written in the same language as the user’s input.
                """
            ),
            llm=self.llm,
            verbose=True,
        )

    def return_response(self, prompt):
        output = self.agent(prompt=prompt)
        return output.model_dump()["input"]

class FetchBot:
    def __init__(self, input_methods):
        self.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", temperature=0)
        react_system_header_str = """

            You are a publication finder magician with access to all of the academic publications on earth.
            You help users find the publication they needed for their research.

            ## Tools
            You have access to one tool. You are responsible for using the tool as many times as you deem
            appropriate with different keywords in order to gather enough information so
            that you can complete the task at hand.

            You have access to the following tool:
            search_publications

            ## Output Format
            To answer the question, please use the following format.

            ```
            Thought: I need to use a tool to help me answer the question.
            Action: tool name (one of {tool_names}) if using a tool.
            Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"keyword": "Kualitas sungai"}})
            ```

            Please ALWAYS start with a Thought.

            Please use a valid JSON format for the Action Input. Do NOT do this {{'keyword': 'Kualitas sungai'}}.

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
            Answer: Sorry, I cannot answer your query. Here's the closest result I can find [your answer here]
            ```
            ## Additional Rules
            - You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.
            - The keywords MUST use the same language as the user input
            - search_publications only accept 1 input that is keywords.

            ## Current Conversation
            Below is the current conversation consisting of interleaving human and assistant messages.
        """

        react_system_prompt = PromptTemplate(react_system_header_str)

        tools = []
        for input_method in input_methods:
            tools.append(FunctionTool.from_defaults(async_fn=input_method))

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=32768)

        self.agent = ReActAgent.from_tools(
            tools,
            chat_mode="react",
            verbose=True,
            memory=self.memory,
            react_system_prompt=react_system_prompt,
            llm=self.llm,
            max_iterations=100,
        )
        self.agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

    def process(self, prompt):
        return self.agent.chat(prompt)

class OutputBot:
    def __init__(self, llm="llama3.1:latest", embedding_model="intfloat/multilingual-e5-large"):
        self.Settings = self.set_setting(llm, embedding_model)
        self.index = self.load_data()
        self.memory = self.create_memory()

    def load_data(self):
        reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
        documents = reader.load_data()
        return VectorStoreIndex.from_documents(documents)

    def set_setting(self, llm, embedding_model):
        self.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = FastEmbedEmbedding(
            model_name=embedding_model, cache_dir="./fastembed_cache")
        return Settings

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def return_response(self, prompt):
        response = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            verbose=True,
            memory=self.memory,
            llm=self.llm,
            context_prompt=(
                "You are a multilingual publication finder magician with access to all of the academic publications on earth."
                "You help users find the publication they needed for their research."
                "Here are the relevant documents for the context:\n"
                "Your task is to provide publication for the user along with the publication's title, link and publication info"
                "If the user ask something else outside of searching publication, try to use your knowledge to answer the user's question while providing the source of that knowledge."
            ),
        ).chat(prompt)
        return response.response