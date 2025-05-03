"""Tests for direct URL crawling functionality."""
import asyncio
import pytest
from othertales.deepresearch.utils import direct_url_crawl, direct_url_tool


@pytest.mark.asyncio
async def test_direct_url_crawl_bs4():
    """Test the direct URL crawling functionality with BeautifulSoup."""
    # Use a reliable test site
    urls = ["https://example.com"]
    results = await direct_url_crawl(urls, use_playwright=False)
    
    # Basic assertions
    assert len(results) == 1
    assert results[0]["query"] == urls[0]
    assert len(results[0]["results"]) == 1
    assert results[0]["results"][0]["title"] == "Example Domain"
    assert "Example Domain" in results[0]["results"][0]["raw_content"]
    assert results[0]["results"][0]["score"] == 1.0


@pytest.mark.asyncio
async def test_direct_url_tool():
    """Test the direct URL tool."""
    # Use a reliable test site
    urls = ["https://example.com"]
    formatted_result = await direct_url_tool(urls, use_playwright=False)
    
    # Basic assertions
    assert isinstance(formatted_result, str)
    assert "URL content extraction results" in formatted_result
    assert "example.com" in formatted_result
    assert "Example Domain" in formatted_result


@pytest.mark.asyncio
async def test_direct_url_crawl_bs4_with_selector():
    """Test the direct URL crawling functionality with BeautifulSoup and CSS selectors."""
    # Use a reliable test site with known elements
    urls = ["https://example.com"]
    results = await direct_url_crawl(
        urls, 
        use_playwright=False,
        extract_selectors=["h1", "p"]
    )
    
    # Basic assertions
    assert len(results) == 1
    assert results[0]["query"] == urls[0]
    assert len(results[0]["results"]) == 1
    
    # Check if extracted content contains the main heading and paragraph
    content = results[0]["results"][0]["content"]
    assert "Example Domain" in content
    assert "for illustrative examples" in content


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for invalid URLs."""
    # Use an invalid URL
    urls = ["https://this-domain-definitely-doesnt-exist-123456789.com"]
    results = await direct_url_crawl(urls, use_playwright=False)
    
    # Basic assertions
    assert len(results) == 1
    assert results[0]["query"] == urls[0]
    assert "error" in results[0]
    assert len(results[0].get("results", [])) == 0


# Skip Playwright tests if not installed
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_INSTALLED = True
except ImportError:
    PLAYWRIGHT_INSTALLED = False


@pytest.mark.asyncio
@pytest.mark.skipif(not PLAYWRIGHT_INSTALLED, reason="Playwright not installed")
async def test_direct_url_crawl_playwright():
    """Test the direct URL crawling functionality with Playwright."""
    # Use a reliable test site
    urls = ["https://example.com"]
    results = await direct_url_crawl(urls, use_playwright=True)
    
    # Basic assertions
    assert len(results) == 1
    assert results[0]["query"] == urls[0]
    assert len(results[0]["results"]) == 1
    assert results[0]["results"][0]["title"] == "Example Domain"
    assert "Example Domain" in results[0]["results"][0]["raw_content"]
    assert results[0]["results"][0]["score"] == 1.0


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_direct_url_crawl_bs4())
    print("BeautifulSoup test passed!")
    
    if PLAYWRIGHT_INSTALLED:
        asyncio.run(test_direct_url_crawl_playwright())
        print("Playwright test passed!")
    else:
        print("Skipping Playwright test as it's not installed.")
    
    asyncio.run(test_direct_url_tool())
    print("Direct URL tool test passed!")
    
    asyncio.run(test_direct_url_crawl_bs4_with_selector())
    print("BeautifulSoup with selector test passed!")
    
    asyncio.run(test_error_handling())
    print("Error handling test passed!")
    
    print("All tests passed!")