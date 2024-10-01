import requests
from scholarly import scholarly

import os
from dotenv import load_dotenv

load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# SerpAPI function tools
async def search_publications_serpapi(keyword: str, language_code: str = "en") -> str:
    """Searches Google Scholar for publications based on the keyword given using SerpAPI. Result: Title, Link, Snippet, Publication info, Cited by count."""
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_scholar",
        "q": keyword,
        "api_key": serpapi_api_key,
        "num": 20,
        "hl": language_code,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'organic_results' not in data:
        return f"No results found for '{keyword}'"

    output = f"# Publication Search Results for '{keyword}'\n"

    for i, result in enumerate(data['organic_results'], start=1):
        title = result.get('title', 'No title')
        link = result.get('link', 'No link available')
        snippet = result.get('snippet', 'No description available')
        pub_info = result.get('publication_info', {}).get('summary', 'No publication info available')
        cited_by_count = result.get('inline_links', {}).get('cited_by', {}).get('total', 'Unknown')

        output += (
            f"   *Title:* {title}\n"
            f"   *Link:* {link}\n"
            f"   *Snippet:* {snippet}\n"
            f"   *Publication info:* {pub_info}\n"
            f"   *Cited by:* {cited_by_count}\n\n"
        )

    return output


# Scholarly function tools
async def search_publications_scholarly(keyword: str, language_code: str = "en") -> str:
    """Searches Google Scholar for publications based on the keyword given using Scholarly package. Result: Title, Link, Abstract, Publication info, Cited by count."""

    # Initialize search query
    search_query = scholarly.search_pubs(keyword)

    # Prepare output
    output = f"# Publication Search Results for '{keyword}'\n"

    # Iterate over search results, limiting to the first 20
    for i in range(20):
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
        output += (
            f"   *Title:* {title}\n"
            f"   *Link:* {link}\n"
            f"   *Abstract:* {abstract}\n"
            f"   *Publication info:* {pub_info}\n"
            f"   *Cited by:* {cited_by_count}\n\n"
        )

    # Return formatted output
    return output