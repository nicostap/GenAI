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
from api import search_publications_serpapi, search_publications_scholarly

nest_asyncio.apply()
st.set_page_config(
    page_title="Publication Finder Bot",
    initial_sidebar_state="expanded",
)

react_context = """
You are a publication finder magician with access to all of the academic publications on earth. You help users find the publication they needed for their research.

## Instruction
DO THIS :
1. GIVING list of publications's title, url, publication info and cited by count.
2. If the user asks a question the you already know the answer to OR the user is making idle banter, just respond without calling any tools.
3. If you don't know the answer, say you DON'T KNOW.
DO NOT DO THIS :
1. SUMMARIZING your findings.
2. NOT giving publications or publication's detail.
3. NOT giving titles.
4. ONLY giving topic.
5. ANSWERING the user's need yourself.
"""

react_system_header_str = """
## Tools
You have access to two tools. You are responsible for experimenting
with the tools using various keyword you deem appropriate to complete the task at hand.
This may require coming up with a new keyword that will get you closer to the right result multiple times.

You have access to the following tools:
search_publications_serpapi, search_publications_scholarly

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
- You MUST NOT summarize your findings.
- You MUST NOT act outside of your context.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""

react_system_prompt = PromptTemplate(react_system_header_str)

Settings.llm = Ollama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434", temperature=0, prompt_key=react_context)

search_publications_tool_serpapi = FunctionTool.from_defaults(async_fn=search_publications_serpapi)
search_publications_tool_scholarly = FunctionTool.from_defaults(async_fn=search_publications_scholarly)
tools = [search_publications_tool_serpapi, search_publications_tool_scholarly]

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
st.title("📚 Publication Finder Bot")
st.markdown("Your personal assistant to help you discover academic publications efficiently.")

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