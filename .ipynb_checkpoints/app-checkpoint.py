import re
import streamlit as st
from scholarly import scholarly
from collections import deque
from bot import InputBot, FetchBot, OutputBot
from llama_index.core import Document
from duckduckgo_search import AsyncDDGS
import requests
from bs4 import BeautifulSoup

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

def scrape_page(url):
    # Make an HTTP request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract specific data (modify this based on your needs)
        title = soup.title.string if soup.title else 'No title found'
        print(f'Title: {title}')
        
        # Example: Extract all paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            print(p.get_text())
    else:
        print(f'Failed to retrieve {url}: {response.status_code}')

async def search_video(keyword: str) -> str:
    """Get video search results from duckduckgo.com based on the keyword given. Based on needs, keywords need to follow a format like the following example:
    1. cats dogs -> Results about cats or dogs
    2. cats and dogs -> Results for exact term "cats and dogs". If no results are found, related results are shown.
    3. cats -dogs -> Fewer dogs in results
    4. cats +dogs -> More dogs in results
    5. cats filetype:pdf -> PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html
    6. dogs site:example.com -> Pages about dogs from example.com
    7. cats -site:example.com -> Pages about cats, excluding example.com
    8. intitle:dogs -> Page title includes the word "dogs"
    9. inurl:cats -> Page url includes the word "cats"
    
    """
    output = f"# Internet Search Result for '{keyword}' video \n"
    search_query = await AsyncDDGS(proxy=None).avideos(keyword, region='wt-wt', safesearch='on', max_results=5)

    for result in search_query:
        # Extracting information from the result
        title = result['title']
        content = result['content']
        description = result['description']
        publisher = result ["publisher"]
        embed_url = result ["embed_url"]
        uploader = result ["uploader"]
                
        #scrape more info from href
        scrape_result = scrape_page(content)

         # Constructing the output
        result = (
            f"   *Title:* {title}\n"
            f"   *Link:* {content}\n"
            f"   *Description:* {description}\n"
            f"   *Content:* {scrape_result}\n"
            f"   *Publisher:* {publisher}\n"
            f"   *Uploader:* {uploader}\n"
            f"   *Embed_URL:* {embed_url} \n\n"
        )

        output += result
        save_to_index(result, title)

    return output

async def search_image(keyword: str) -> str:
    """Get image search results from duckduckgo.com based on the keyword given. Based on needs, keywords need to follow a format like the following example:
    1. cats dogs -> Results about cats or dogs
    2. cats and dogs -> Results for exact term "cats and dogs". If no results are found, related results are shown.
    3. cats -dogs -> Fewer dogs in results
    4. cats +dogs -> More dogs in results
    5. cats filetype:pdf -> PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html
    6. dogs site:example.com -> Pages about dogs from example.com
    7. cats -site:example.com -> Pages about cats, excluding example.com
    8. intitle:dogs -> Page title includes the word "dogs"
    9. inurl:cats -> Page url includes the word "cats"
    
    """
    output = f"# Internet Search Result for '{keyword}' image \n"
    
    search_query = await AsyncDDGS(proxy=None).aimages(keyword, region='wt-wt', safesearch='on', max_results=5)

    for result in search_query:
        # Extracting information from the result
        title = result['title']
        url = result['url']
        image = result['image']

        #scrape more info from href
        scrape_result = scrape_page(url)

         # Constructing the output
        result = (
            f"   *Title:* {title}\n"
            f"   *Image URL:* {image}\n"
            f"   *Link:* {link}\n"
            f"   *Content:* {scrape_result}\n\n"
        )

        output += result
        save_to_index(result, title)

    return output


async def search_internet(keyword: str) -> str:
    """Get search results from duckduckgo.com based on the keyword given. Based on needs, keywords need to follow a format like the following example:
    1. cats dogs -> Results about cats or dogs
    2. cats and dogs -> Results for exact term "cats and dogs". If no results are found, related results are shown.
    3. cats -dogs -> Fewer dogs in results
    4. cats +dogs -> More dogs in results
    5. cats filetype:pdf -> PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html
    6. dogs site:example.com -> Pages about dogs from example.com
    7. cats -site:example.com -> Pages about cats, excluding example.com
    8. intitle:dogs -> Page title includes the word "dogs"
    9. inurl:cats -> Page url includes the word "cats"
    
    """
    output = f"# Internet Search Result for '{keyword}' text \n"

    search_query = await AsyncDDGS(proxy=None).atext(keyword, region='wt-wt', safesearch='on', max_results=5)

    for result in search_query:
        # Extracting information from the result
        title = result['title']
        href = result['href']
        body = result['body']

        #scrape more info from href
        scrape_result = scrape_page(href)

         # Constructing the output
        result = (
            f"   *Title:* {title}\n"
            f"   *Link:* {href}\n"
            f"   *Body:* {body}\n"
            f"   *Detail:* {scrape_result}\n\n"
        )

        output += result
        save_to_index(result, title)
   
    return output
    

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

# Function to check if a link is an image
def is_image_link(url):
    return url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))

# Function to check if a link is a YouTube video link
def is_video_link(url):
    return re.match(r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)', url) is not None


if "fetch_bot" not in st.session_state:
    st.session_state.fetch_bot = FetchBot([search_publications, search_internet, search_video, search_image])

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

        # Check if the link is an image or video
        if is_image_link(link):
            # Display image link
            st.image(link, caption=match.group("domain"), use_column_width=True)
        elif is_video_link(link):
            # Display video link
            st.video(link)
        else:
            # Display as a regular link button
            st.link_button(match.group("domain"), link)
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



