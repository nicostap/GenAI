from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import (
    PromptTemplate,
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.empty import EmptyIndex
from httpx import Timeout
import qdrant_client
import streamlit as st
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import SummaryIndex, Document
from llama_index.core import PropertyGraphIndex

class InputBot:
    def __init__(self):
        self.llm = Ollama(
            model="qwen2.5-coder:32b-instruct-q5_1",
            base_url="http://127.0.0.1:11434",
            temperature=0,
            request_timeout=3000.0,
        )
        self.header_prompt =  """
                ## Instruction
                1. Receive the input query
                2. Identify the language of the conversation from the input query, determine the language the query uses.
                3. Understand the input query, summarize what the query wants to know or ask into one or more topics. Decice which media format the user is seeking, if it's not specififed the media format is "webs"
                4. Response format : "["Give me {videos/webs/pictures/publications} about {topic 1}", "Give me {videos/webs/pictures/publications} about {topic 2}", ...]"

                Below is the input query you will receive and process:
            """


    def return_response(self, message, prompt):
        prompt = f"{self.header_prompt}\n{prompt}"
        input_list = message.copy()
        input_list.append({"role": "user", "content": prompt})
        chat_messages = list(
            map(
                lambda item: ChatMessage(role=item["role"], content=item["content"]),
                input_list,
            )
        )
        response = self.llm.chat(chat_messages)
        print(response.message.content)
        return response.message.content[
            response.message.content.rfind("[") : response.message.content.rfind("]")
            + 1
        ]


class FetchBot:
    def __init__(self, input_methods):
        self.llm = Ollama(
            model="qwen2.5-coder:32b-instruct-q5_1", base_url="http://127.0.0.1:11434", temperature=0,
            request_timeout=3000.0
        )
        react_system_header_str = """

            You are designed to be able to search and access any information source such as publication papers.
            Your role is to answer any request that the user have using the tools available.

            ## Tools
            You have access to a wide variety of tools. You are responsible for using
            the tools in any sequence you deem appropriate to complete the task at hand.
            This may require breaking the task into subtasks and using different tools
            to complete each subtask.

            You have access to the following tool:
            search_publications, search_internet, search_video, search_image

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
            You should keep repeating the above format until you have more than enough information
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
            - ALL of the tools ONLY accept ONE (1) input that is a string called keyword.
            - keyword MUST be from the same language as what the user is using.
            - If you observe ERROR, you should STOP and GIVE UP.

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
        # Instead of evaluating the input, check if it is already a valid list or string
        if isinstance(prompt_string, str):
            # Wrap the string in a list if it's a single string
            prompts = [prompt_string]
        else:
            # Otherwise, assume it is a list of prompts
            prompts = prompt_string
        for prompt in prompts:
            return self.agent.chat(prompt)


class OutputBot:
    def __init__(self, index=None, system_prompt=None):
        self.system_prompt = system_prompt or """
            You are a multilingual source finder magician with access to all of the information sources on earth.

            ## Instruction
            1. You help users find the source they needed for an information.
            2. Your task is to provide source's TITLE for the user along with the source's DETAIL (except for title) from your context.
            3. The source's TITLE and DETAIL must be from the SAME DOCUMENT.
            4. ALWAYS provide a link as a source for your answer.
            5. Please provide your response in a concise, well-structured, and visually clear response using bullet points and bold headings.

            Here are the relevant documents filled with title, links and other details for your context: {context_str}
            Answer the user quere here : {query_str} by following the instruction above.  
        """
        self.Settings = self.set_setting()
        self.index = index or self.load_data()
        self.create_chat_history()
        self.refresh_memory()

    def load_data(self):
        reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
        documents = reader.load_data()
        try:
            client = qdrant_client.QdrantClient(
                url=st.secrets["qdrant"]["connection_url"],
                api_key=st.secrets["qdrant"]["api_key"],
                timeout=Timeout(timeout=5.0),
            )
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="Documents"
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
        except:
            return VectorStoreIndex.from_documents(documents)

    def set_setting(self):
        self.llm = Ollama(model="qwen2.5-coder:32b-instruct-q5_1", base_url="http://127.0.0.1:11434", temperature=0, request_timeout=3000.0)
        Settings.embed_model = OllamaEmbedding(
            base_url="http://127.0.0.1:11434",
            model_name="mxbai-embed-large:latest"
        )
        return Settings

    def set_chat_history(self, messages):
        self.chat_history = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in messages
        ]
        self.chat_store.store = {"chat_history": self.chat_history}
        self.refresh_memory()

    def create_chat_history(self):
        self.chat_store = SimpleChatStore()

    def refresh_memory(self):
        self.memory = ChatMemoryBuffer.from_defaults(
            chat_store=self.chat_store,
            chat_store_key="chat_history",
            token_limit=16000
        )

    def return_response(self, prompt):
        text_qa_template = PromptTemplate(self.system_prompt)
        response = self.index.as_query_engine(
            # verbose=True,
            memory=self.memory,
            llm=self.llm,
            text_qa_template=text_qa_template,
        ).query(prompt)
        return response.response

    def insert_doc(self, doc):
        self.index.insert(doc)

    def delete_doc(self, doc_id):
        self.index.delete_ref_doc(
            doc_id, delete_from_docstore=True
        )


class AnswerBot(OutputBot):
    def __init__(self, index):
        system_prompt = """
                You are a multilingual answer giving magician with access to all of the webpage information on earth.
                ## Instruction
                1. You help users answer their question.
                2. Your task is to search for documents with type "Webpage" from your context and answer to the user's question based on the documents's content.
                3. Provide the source document's TITLE and LINK as well.
                4. The source document's TITLE and LINK must be from the SAME DOCUMENT.
                5. ALWAYS provide a link as a source for your answer.

                Here are the relevant documents filled with title, links and other details for your context: {context_str}
                Answer the user quere here : {query_str} by following the instruction above.
        """
        super().__init__(index, system_prompt)


class TriageBot:
    def __init__(self):
        self.llm = Ollama(
            model="qwen2.5-coder:32b-instruct-q5_1",
            base_url="http://127.0.0.1:11434",
            temperature=0,
            request_timeout=3000.0
        )
        self.header_prompt =  """
                ## Instruction
                1. Receive and process the latest input query from user
                2. Response with 1 if the user wants to search for sources (videos/webs/pictures/publications/sources)
                3. Response with 0 if the user wants to ask questions.
                4. Response format : "[0]" or "[1]"

                Below is the input query you will receive and process:
            """

    def return_response(self, message, prompt):
        prompt = f"{self.header_prompt}\n{prompt}"
        input_list = message.copy()
        input_list.append({"role": "user", "content": prompt})
        chat_messages = list(
            map(
                lambda item: ChatMessage(role=item["role"], content=item["content"]),
                input_list,
            )
        )
        response = self.llm.chat(chat_messages)
        print(response.message.content)
        return response.message.content[
            response.message.content.rfind("[") : response.message.content.rfind("]")
            + 1
        ]
