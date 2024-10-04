from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex, Document, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import re
import ast

class InputBot:
    def __init__(self):
        self.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", temperature=0, )

        self.header_prompt =  """
                Do these step below no matter what language the user is speaking
                ## Instruction / Instruksi
                1. Receive the user prompt. / Terima prompt pengguna.
                1. Identify the language of the conversation from user's prompt, determine the language the user is speaking. / Tentukan bahasa yang digunakan pengguna.
                2. Understand the user's prompt, summarize what the user wants to know or ask into one or more topics. / Pahami prompt pengguna dan rangkumkan menjadi topik.
                3. Return the topics in the user's language in the format of python list such as ["Give me sources about {topic 1}", "Give me sources about {topic 2}", ...] / Berikan respon topik tersebut di menggunakan bahasa pengguna dalam format list python seperti ["Berikan saya sumber tentang {topik 1}", "Berikan saya sumber tentang {topik 2}", ...]
            """

    def return_response(self, prompt):
        response = self.llm.complete(f"{self.header_prompt}\n{prompt}")
        print(response.text)
        return response.text[response.text.rfind("[") : response.text.rfind("]") + 1]

class FetchBot:
    def __init__(self, input_methods):
        self.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", temperature=0)
        react_system_header_str = """

            You are designed to be able to search and access any information source such as publication papers.
            Your role is to answer any request that the user have using the tools available.

            ## Tools
            You have access to a wide variety of tools. You are responsible for using
            the tools in any sequence you deem appropriate to complete the task at hand.
            This may require breaking the task into subtasks and using different tools
            to complete each subtask.

            You have access to the following tool:
            search_publications

            ## Output Format
            To answer the question, please use the following format.

            ```
            Thought: I need to use a tool to help me answer the question.
            Action: tool name (one of {tool_names}) if using a tool.
            Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"keyword": "Kualitas sungai"}}
            ```

            Please ALWAYS start with a Thought.

            Please use a valid JSON format for the Action Input. Do NOT do this {{'keyword': 'Kualitas sungai'}}

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
            - search_publications ONLY accept ONE (1) input that is keywords.
            - keywords MUST be from the same language as what the user is using.

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

    def process(self, prompt_string):
        prompts = ast.literal_eval(prompt_string)
        for prompt in prompts:
            return self.agent.chat(prompt)

class OutputBot:
    def __init__(self, llm="llama3.1:latest", embedding_model="intfloat/multilingual-e5-large"):
        self.system_prompt = """
                You are a multilingual source finder magician with access to all of the information sources on earth.

                ## Instruction
                1. You help users find the source they needed for an information.
                2. Your task is to provide source's title for the user along with the source's detail (except for title) from your context.
                3. If the user ask something else outside of searching publication, try to use your context to answer the user's question while providing the source or link of that context.

                Here are the relevant documents filled with title, links and other details for your context: {context_str}
                Answer the user quere here : {query_str} by following the instruction above.  
        """
        self.Settings = self.set_setting(llm, embedding_model)
        self.index = self.load_data()
        self.create_chat_history()
        self.refresh_memory()

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
        self.refresh_memory()

    def create_chat_history(self):
        self.chat_store = SimpleChatStore()

    def refresh_memory(self):
        self.memory = ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def return_response(self, prompt):
        text_qa_template = PromptTemplate(self.system_prompt)
        response = self.index.as_query_engine(
            verbose=True,
            memory=self.memory,
            llm=self.llm,
            text_qa_template=text_qa_template,
        ).query(prompt)
        return response.response