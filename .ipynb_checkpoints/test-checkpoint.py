from crawl4ai import AsyncWebCrawler
import asyncio

async def open_link(link: str) -> str:
    """Open a link and scrap the content."""
    async with AsyncWebCrawler(verbose=True) as crawler:
        scraps = await crawler.arun(url=link)
        result = scraps.markdown[:1000]
        print(result)


if __name__ == "__main__":
    asyncio.run(open_link("https://www.nbcnews.com/business"))