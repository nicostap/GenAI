import re
import streamlit as st
from scholarly import scholarly
from collections import deque
from bot import InputBot, FetchBot, OutputBot, AnswerBot, TriageBot
from llama_index.core import Document
from duckduckgo_search import AsyncDDGS
import requests
from bs4 import BeautifulSoup
import time
from crawl4ai import WebCrawler
from PIL import Image
from io import BytesIO
from ollama import generate
import json

# Toggle Feature
SCREENSHOOT_WEB = False

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
            "content": "Hello! What information can I help you find today?",
        }
    ]
if "input_bot" not in st.session_state:
    st.session_state.input_bot = InputBot()
if "output_bot" not in st.session_state:
    st.session_state.output_bot = OutputBot()
if "answer_bot" not in st.session_state:
    st.session_state.answer_bot = AnswerBot(st.session_state.output_bot.index)
if "triage_bot" not in st.session_state:
    st.session_state.triage_bot = TriageBot()
if "links" not in st.session_state:
    st.session_state.links = []
if "crawler" not in st.session_state:
    st.session_state.crawler = WebCrawler()
    st.session_state.crawler.warmup()


# Special dependencies
def title_to_variable(title):
    title = title.lower()
    title = title.replace(" ", "_")
    title = re.sub(r"[^a-z0-9_]", "", title)
    return title


def is_url_dead(url):
    try:
        # Check with HEAD request
        head_response = requests.head(url, timeout=5)
        if head_response.status_code == 404 or head_response.status_code >= 500:
            return True

        # Check with GET request
        get_response = requests.get(url, timeout=5)
        if "404" in get_response.text.lower() or "not found" in get_response.text.lower():
            return True
        return False
    except Exception:
        return False


def is_image_dead(url):
    if not is_image_link(url):
        return True

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 404 or response.status_code >= 500:
            return True

        # Check Content-Type header
        content_type = response.headers.get("Content-Type", "").lower()
        if not content_type.startswith("image/"):
            return True

        image_data = response.content
        try:
            Image.open(BytesIO(image_data)).verify()
        except Exception:
            return True

        return False
    except Exception:
        return False


def is_video_dead(url):
    if not is_video_link(url):
        return True

    try:
        if "youtu.be" in url:
            video_id = url.split("/")[-1]
            url = f"https://www.youtube.com/embed/{video_id}"
        elif "watch?v=" in url:
            video_id = url.split("watch?v=")[-1]
            url = f"https://www.youtube.com/embed/{video_id}"

        response = requests.get(url, timeout=10)
        if response.status_code == 404 or response.status_code >= 500:
            return True

        if "UNPLAYABLE" in response.text:
            return True

        return False
    except Exception:
        return False


def save_to_index(content, indentifier, url, url_checker=None):
    url_checker = url_checker or is_url_dead
    if url_checker(url):
        return

    doc_id = title_to_variable(indentifier)
    metadata = {"filename": doc_id, "timestamp": int(time.time())}
    metadata.update(content)
    doc = Document(
        text=json.dumps(content),
        id_=doc_id,
        metadata=metadata,
    )
    st.session_state.output_bot.insert_doc(doc)
    st.session_state.track_index.append(doc_id)

    if len(st.session_state.track_index) > 100:
        old_doc_id = st.session_state.track_index.popleft()
        st.session_state.output_bot.delete_doc(old_doc_id)


def scrape_page(url):
    # Make an HTTP request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract specific data (modify this based on your needs)
        title = soup.title.string if soup.title else "No title found"
        print(f"Title: {title}")

        # Example: Extract all paragraphs
        paragraphs = soup.find_all("p")
        result = ""
        for p in paragraphs:
            result += p.get_text() + "\n"
        return result
    else:
        print(f"Failed to retrieve {url}: {response.status_code}")
        return "Failed to fetch"


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
    search_query = await AsyncDDGS(proxy=None).avideos(
        keyword, region="wt-wt", safesearch="on", max_results=5
    )

    for result in search_query:
        # Extracting information from the result
        title = result["title"]
        content = result["content"]
        description = result["description"]
        publisher = result["publisher"]
        embed_url = result["embed_url"]
        uploader = result["uploader"]

        # Scrape more info from href
        scrape_result = scrape_page(content)

        # Constructing the output
        result = {
            "Type": "Video",
            "Title": title,
            "Link": content,
            "Description": description,
            "Content": scrape_result,
            "Publisher": publisher,
            "Uploader": uploader,
            "Embed_URL": embed_url,
        }
        output += json.dumps(result)
        save_to_index(result, title, embed_url, is_video_dead)

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

    search_query = await AsyncDDGS(proxy=None).aimages(
        keyword, region="wt-wt", safesearch="on", max_results=5
    )

    for result in search_query:
        # Extracting information from the result
        title = result["title"]
        image = result["image"]

        if not is_image_link(image):
            continue

        # Scrape more info from href
        content = annotate_image(image)

        # Constructing the output
        result = {
            "Type": "Image",
            "Title": title,
            "Link": image,
            "Content": content,
        }
        output += json.dumps(result)
        save_to_index(result, title, image, is_image_dead)

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

    search_query = await AsyncDDGS(proxy=None).atext(
        keyword, region="wt-wt", safesearch="on", max_results=5
    )

    for result in search_query:
        # Extracting information from the result
        title = result["title"]
        href = result["href"]
        body = result["body"]

        # Scrape more info from href
        scrape_result = scrape_page(href)

        # Constructing the output
        result = {
            "Type": "Webpage",
            "Title": title,
            "Link": href,
            "Body": body,
            "Detail": scrape_result,
        }
        output += json.dumps(result)
        save_to_index(result, title, href)

    return output


async def search_publications(keyword: str) -> str:
    """Searches Google Scholar for publications based on the keyword given."""

    # Initialize search query
    search_query = scholarly.search_pubs(keyword)

    # Prepare output
    output = f"# Publication Search Results for '{keyword}'\n"

    # Iterate over search results, limiting to the first 4
    for i in range(10):
        try:
            result = next(search_query)
        except StopIteration:
            break  # No more results

        # Extracting information from the result
        title = result["bib"].get("title", "No title")
        author = ", ".join(result["bib"].get("author", ["Unknown author"]))
        pub_year = result["bib"].get("pub_year", "Unknown year")
        venue = result["bib"].get("venue", "Unknown venue")
        pub_info = f"{author}, {pub_year}, {venue}"
        abstract = result["bib"].get("abstract", "No description available")
        cited_by_count = result.get("num_citations", "Unknown")
        link = result.get("pub_url", "No link available")

        # Constructing the output
        result = {
            "Type": "Publication",
            "Title": title,
            "Link": link,
            "Abstract": abstract,
            "Publication_info": pub_info,
            "Cited_by": cited_by_count,
        }
        output += json.dumps(result)
        save_to_index(result, title, link)

    return output


# Function to check if a link is an image
def is_image_link(url):
    return url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))


# Function to check if a link is a YouTube video link
def is_video_link(url):
    return (
        re.match(
            r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)",
            url,
        )
        is not None
    )


if "fetch_bot" not in st.session_state:
    st.session_state.fetch_bot = FetchBot(
        [search_publications, search_internet, search_video, search_image]
    )


def annotate_image(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        image_bytes = BytesIO(response.content).getvalue()
    except requests.RequestException:
        return ''

    if not image_bytes:
        return ''

    context = ''
    for response in generate(
        model='llava:13b-v1.6',
        prompt='Describe this image you are seeing it in person, do not say the word "image" :',
        images=[image_bytes],
        stream=True
    ):
        context += response['response']
    return context


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
        match = re.match(
            r"(https?://)?(www\d?\.)?(?P<domain>[\w\.-]+\.\w+)(/\S*)?", link
        )
        # Check if the link is an image or video
        if is_image_link(link):
            # Display image link
            st.image(link, caption=match.group("domain"), use_column_width=True)
        elif is_video_link(link):
            # Display video link
            st.video(link)
        else:
            # Display as a regular link button
            if SCREENSHOOT_WEB:
                image_result = st.session_state.crawler.run(url=link, screenshot=True)
                image_data = image_result.screenshot
                if image_data is not None:
                    st.markdown(
                        f"""
                            <a href="{link}" target="_blank">
                                <div style="width: 100%; border-radius: 20px; max-height: 200px; overflow:hidden;">
                                    <img src="data:image/png;base64,{image_data}" alt="Screenshot" style="width: 101%;">
                                </div>
                            </a>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.link_button(match.group("domain"), link)
            else:
                st.link_button(match.group("domain"), link)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            fetch_query = st.session_state.input_bot.return_response(
                st.session_state.messages, prompt
            )
            thought_process = st.session_state.fetch_bot.process(fetch_query)
            triage_result = st.session_state.triage_bot.return_response(st.session_state.messages, prompt)

            response = ""
            if "0" in triage_result:
                response = st.session_state.answer_bot.return_response(prompt)
                response += "\n\nAnswered by AnswerBot"
            elif "1" in triage_result:
                response = st.session_state.output_bot.return_response(prompt)
                response += "\n\nAnswered by OutputBot"
            else:
                response = "Sorry I could not understand your request"

            urls = re.findall(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                response,
            )
            st.session_state.links.clear()
            st.session_state.links.extend(urls)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()