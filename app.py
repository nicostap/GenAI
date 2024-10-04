import re
import streamlit as st
from scholarly import scholarly
from collections import deque
from bot import InputBot, FetchBot, OutputBot
from llama_index.core import Document

# Initialize everything
st.set_page_config(
    page_title="Source Bot",
)
if "memory" not in st.session_state:
    st.session_state.track_index = deque()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you find academic publications today?",
        }
    ]
if "input_bot" not in st.session_state:
    st.session_state.input_bot = InputBot()
if "output_bot" not in st.session_state:
    st.session_state.output_bot = OutputBot()
if "links" not in st.session_state:
    st.session_state.links = []

# Special dependencies
def title_to_variable(title):
    title = title.lower()
    title = title.replace(" ", "_")
    title = re.sub(r'[^a-z0-9_]', '', title)
    return title

def save_to_index(content, indentifier):
    doc_id = title_to_variable(indentifier)
    doc = Document(text=content, id_=doc_id)
    st.session_state.output_bot.index.insert(doc)
    st.session_state.track_index.append(doc_id)

    # If context is full, pop old ones
    if len(st.session_state.track_index) > 200:
        old_doc_id = st.session_state.track_index.popLeft()
        st.session_state.output_bot.index.delete_ref_doc(old_doc_id, delete_from_docstore=True)

async def search_publications(keyword: str) -> str:
    """Searches Google Scholar for publications based on the keyword given."""

    # Initialize search query
    search_query = scholarly.search_pubs(keyword)

    # Prepare output
    output = f"# Publication Search Results for '{keyword}'\n"

    # Iterate over search results, limiting to the first 4
    for i in range(4):
        try:
            result = next(search_query)
        except StopIteration:
            break  # No more results

        # Extracting information from the result
        title = result['bib'].get('title', 'No title')
        author = ", ".join(result['bib'].get('author', ['Unknown author']))
        pub_year = result['bib'].get('pub_year', 'Unknown year')
        venue = result['bib'].get('venue', 'Unknown venue')
        pub_info = f"{author}, {pub_year}, {venue}"
        abstract = result['bib'].get('abstract', 'No description available')
        cited_by_count = result.get('num_citations', 'Unknown')
        link = result.get('pub_url', 'No link available')

        # Constructing the output
        result = (
            f"   *Title:* {title}\n"
            f"   *Link:* {link}\n"
            f"   *Abstract:* {abstract}\n"
            f"   *Publication info:* {pub_info}\n"
            f"   *Cited by:* {cited_by_count}\n\n"
        )
        output += result
        save_to_index(result, title)

    return output

if "fetch_bot" not in st.session_state:
    st.session_state.fetch_bot = FetchBot([search_publications])

# Main Program
st.title("📚 SourceBot")
st.markdown("Find any sources of information")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.session_state.output_bot.set_chat_history(st.session_state.messages)

st.sidebar.title("🔗 Saved Links")
with st.sidebar.container():
    if len(st.session_state.links) == 0:
        st.markdown("No saved links yet")
    for link in st.session_state.links:
        match = re.match(r"(https?://)?(www\d?\.)?(?P<domain>[\w\.-]+\.\w+)(/\S*)?", link)
        st.link_button(match.group("domain"), link)

# Chat input from user
if prompt := st.chat_input("What is up?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            fetch_query = st.session_state.input_bot.return_response(prompt)
            print(fetch_query)

            thought_process = st.session_state.fetch_bot.process(fetch_query)

            response = st.session_state.output_bot.return_response(prompt)
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response)
            st.session_state.links.clear()
            st.session_state.links.extend(urls)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()