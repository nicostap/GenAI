import streamlit as st
import nest_asyncio
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from typing import Optional
from api import search_publications_serpapi

nest_asyncio.apply()
st.set_page_config(
    page_title="Publication Finder Bot",
    layout="wide",
)

system_prompt = """
## Instruction
If the user asks a question the you already know the answer to OR the user is making idle banter, just respond without calling any tools.
DO THIS :
1. Giving list of publications's title and year.
2. Giving full detail of a publication including year, link, author, cited count, etc.
DO NOT DO THIS :
1. Summarizing.
2. Doing things outside of giving publications or publication's detail.
3. Not giving titles of publication.
4. Only giving topic.
5. Not giving any details.
"""

react_context = """
You are a publication finder magician with access to all of the academic publications on earth. You help users find the publications they needed for their research.
If you don't know the answer, say you DON'T KNOW.

## Instruction
DO THIS :
1. Giving list of publications's title. author and year.
2. Giving additional detail of a publication if the user requested it.
DO NOT DO THIS :
1. Summarizing.
2. Doing things outside of giving publications or publication's detail.
3. Not giving titles of publication.
4. Only giving topic.
5. Not giving any details.
"""

react_system_header_str = """
## Tools
You have access to a tool. You are responsible for using
the tool with various keyword you deem appropriate to complete the task at hand.
This may require coming up with a new keyword that will get you closer to the right result.

You have access to the following tools:
search_publications_serpapi

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"keyword": "Kualitas sungai", "language_code": "id"}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'keyword': 'Kualitas sungai', 'language_code': 'id'}}.

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
- Use the language_code from the language the user is speaking.
- If you can not answer the query, provide the closest answer you can find.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""

react_system_prompt = PromptTemplate(react_system_header_str)

Settings.llm = Ollama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434", system_prompt=system_prompt, temperature=0)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

search_publications_tool = FunctionTool.from_defaults(async_fn=search_publications_serpapi)
tools = [search_publications_tool]

chat_engine = ReActAgent.from_tools(
    tools,
    chat_mode="react",
    verbose=True,
    react_system_prompt=react_system_prompt,
    llm=Settings.llm,
    context=react_context,
    max_iterations=100,
)

# Main Program
st.title("Publication Finder Bot")

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you find academic publications today?",
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input from user
if prompt := st.chat_input("What are you looking for?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            response_stream = chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response_stream.response}
    )
