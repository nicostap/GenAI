import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Optional
from llama_index.core import PromptTemplate

import nest_asyncio
nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

system_prompt = """
You are a multi-lingual career advisor expert who has knowledge based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

You primary job is to help students find courses related to their interests from the Kampus Merdeka Platform.

When a user is asking about possible activities, you should at least mention the name of the activity, the name of the mitra, and the location. Use numbered lists to show the user possible results. Show the users possible activities that they can take before asking which one they are interested in.

If the user asks about the activity in detail, you should at least mention:
1. The mitra which hosts the activity and the name of the activity.
2. A summary of what the activity is about.
3. When it is held.
4. Whether it is eligible to be converted into university credits, if so, how many.

Here is a short example:
User: I would like to study Machine Learning.
Assistant: Sure! I found a few results related to Machine Learning:
1. Bangkit Academy, held by Google, Online.
2. AI & Machine Learning, held by MMT at Jakarta.
User: I'd like to learn more about Bangkit Academy.
Assistant: Sure! <elaborate about bangkit academy here>

"""

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

import sys

import logging
import requests

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = Ollama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434", system_prompt=system_prompt, temperature=0)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

# Main Program
st.title("RAG Test")

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! Mau tahu apa tentang Studi Independen?"}
    ]

# Declare Tools
# function tools
async def search_studi_independen(keyword: str, location_key: Optional[str]) -> list[str]:
    """Searches the Studi Independen database for matching studi independen entries. Keyword should be one or two relevant words. location_key should be 'Surabaya' or 'Online' or empty."""
    r = requests.get("https://api.kampusmerdeka.kemdikbud.go.id/studi/browse/activity", {
        "offset": 0,
        "limit": 50,
        "location_key": location_key,
        "keyword": keyword,
        "sector_id": None,
        "sort_by": "published_time",
        "order": "desc"
    })

    data = r.json()
    output = f"# Course Search Results for '{keyword}'"

    for d in data["data"]:
        output += f"""
Activity Name: {d['name']}
Type: {d['activity_type']}
Location: {d['location']}
Mitra: {d['mitra_name']}
Activity Id: {d['id']}

"""
    return output


async def get_studi_independen_activity_detail(activity_id: str) -> str:
    """Provides detailed information regarding the studi independen activity."""
    r = requests.get(f"https://api.kampusmerdeka.kemdikbud.go.id/studi/browse/activity/{activity_id}")

    data = r.json()["data"]
    return f"""
Activity Name: {data["name"]}
Activity Type: {data["activity_type"]}
Location: {data["location"]}

Description:
{data["description"]}

Requirements:
{data["requirement"]}
    """


search_studi_independen_tool = FunctionTool.from_defaults(async_fn=search_studi_independen)
get_studi_independen_activity_detail_tool = FunctionTool.from_defaults(async_fn=get_studi_independen_activity_detail)

tools = [search_studi_independen_tool, get_studi_independen_activity_detail_tool]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Mau tahu apa tentang Studi Independen?"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=32768)
    st.session_state.chat_engine = ReActAgent.from_tools(
        tools,
        chat_mode="react",
        verbose=True,
        memory=memory,
        react_system_prompt=react_system_prompt,
        # retriever=retriever,
        llm=Settings.llm
    )

    print(st.session_state.chat_engine.get_prompts())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream.response})
