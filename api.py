import requests
from scholarly import scholarly

import os
from dotenv import load_dotenv

load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# SerpAPI function tools
async def search_publications_serpapi(keyword: str, language_code: str = "en") -> str:
    """Searches Google Scholar for publications based on the keyword given."""
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
            f"{i}. *Title:* {title}\n"
            f"   *Link:* {link}\n"
            f"   *Snippet:* {snippet}\n"
            f"   *Publication info:* {pub_info}\n"
            f"   *Cited by:* {cited_by_count}\n\n"
        )

    return output


# Scholarly function tools
async def search_publications_scholarly(keyword: str) -> str:
    """Searches Google Scholar for publications related to the keyword which only correlates to titles."""
    search_query = scholarly.search_pubs(keyword)
    output = f"# Publication Search Results for '{keyword}'\n"

    for i, result in enumerate(search_query, start=1):
        paper_info = result["bib"]
        title = paper_info.get("title", "No title")
        authors = ", ".join(paper_info.get("author", []))
        pub_year = paper_info.get("pub_year", "Unknown")
        venue = paper_info.get("venue", "Unknown venue")

        output += f"{i}. **{title}** by {authors} ({pub_year}) - {venue}\n\n"
        if i >= 10:
            break

    return output