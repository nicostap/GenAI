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
from httpx import Timeout
import ast
import qdrant_client
import streamlit as st


class InputBot:
    def __init__(self):
        self.llm = Ollama(
            model="llama3.1:latest",
            base_url="http://127.0.0.1:11434",
            temperature=0,
        )
        self.header_prompt = """
                Do these step below no matter what language the user is speaking
                ## Instruction in English
                1. Receive the latest user prompt.
                2. Determine the language from the latest prompt.
                3. Understand the latest prompt, summarize what the user wants to know or ask into one or more topics.
                4. Return the topics as a response (ONLY IN A SINGLE LANGUAGE USING THE LATEST'S PROMPT LANGUAGE) in the format of python list like this ["Give me sources about {topic 1}", "Give me sources about {topic 2}", ...]
                ## Example 1
                User : I want to learn how to cook mussel
                Assistant : ["Give me video about how to cook mussel", "Give me sources about healthy ways of eating mussel"]

                ## Example 2
                User: Saya ingin belajar cara masak kerang
                Assistant: ["Berikan saya video tentang cara memasak kerang", "Berikan saya sumber tentang cara sehat memasak kerang"]
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
            model="llama3.1:latest", base_url="http://127.0.0.1:11434", temperature=0
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
        prompts = ast.literal_eval(prompt_string)
        # prompts = re.findall(r'\"(.*?)\"', prompt_string)
        for prompt in prompts:
            return self.agent.chat(prompt)


class OutputBot:
    def __init__(
        self, llm="llama3.1:latest", embedding_model="intfloat/multilingual-e5-large"
    ):
        self.system_prompt = """
                You are a multilingual source finder magician with access to all of the information sources on earth.

                ## Instruction
                1. You help users find the source they needed for an information.
                2. Your task is to provide source's TITLE for the user along with the source's DETAIL (except for title) from your context.
                3. The source's TITLE and DETAIL must be from the SAME DOCUMENT.
                4. If the user ask something else outside of searching publication, try to use your context to answer the user's question while providing the source or link of that context.
                5. You can also find and provide another related sources to the response. The related sources can be article, video, or image link.
                6. ALWAYS provide a link as a source for your answer.
                7. Please provide a concise, well-structured, and visually clear response using bullet points, bold headings, and short paragraphs where appropriate.

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

    def set_setting(self, llm, embedding_model):
        self.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
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
            verbose=True,
            memory=self.memory,
            llm=self.llm,
            text_qa_template=text_qa_template,
        ).query(prompt)
        return response.response