"""
URL Scraper Module for Fake News Detection
Extracts article content from news URLs using BeautifulSoup + Playwright
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import playwright (optional dependency)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None


class URLScraper:
    """
    Hybrid URL scraper with BeautifulSoup + Playwright fallback
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    async def scrape(self, url: str) -> Dict:
        """Main scraping method with fallback (ASYNC)"""
        if not self._is_valid_url(url):
            return {
                "success": False,
                "text": None,
                "method": None,
                "error": "Invalid URL format",
                "metadata": {}
            }
        
        logger.info(f"Attempting to scrape: {url}")
        logger.info(f"Playwright available: {PLAYWRIGHT_AVAILABLE}")
        
        # Try BeautifulSoup first (fast)
        result = self._basic_scrape(url)
        if result["success"]:
            logger.info(f"Scraping successful for: {url}")
            return result
        
        # Try Playwright fallback
        if PLAYWRIGHT_AVAILABLE:
            logger.info(f"Basic scraping failed, trying Playwright for: {url}")
            result = await self._playwright_scrape(url)
            if result["success"]:
                logger.info(f"Playwright scraping successful for: {url}")
                return result
        else:
            logger.warning("Playwright not available, cannot use fallback")
        
        logger.error(f"All scraping methods failed for: {url}")
        return {
            "success": False,
            "text": None,
            "method": None,
            "error": "Cannot access this URL. The website may be blocking automated access or requires authentication.",
            "metadata": {}
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False
    
    def _basic_scrape(self, url: str) -> Dict:
        """BeautifulSoup scraping"""
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'form']):
                element.decompose()
            
            article_text = None
            title_text = ""
            
            # Strategy 1: <article> tag
            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                article_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Strategy 2: main content
            if not article_text or len(article_text) < 200:
                main_content = soup.find('main') or soup.find('div', {'role': 'main'})
                if main_content:
                    paragraphs = main_content.find_all('p')
                    text = ' '.join([p.get_text().strip() for p in paragraphs])
                    if len(text) > len(article_text or ''):
                        article_text = text
            
            # Strategy 3: all paragraphs
            if not article_text or len(article_text) < 200:
                paragraphs = soup.find_all('p')
                paragraphs = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                article_text = ' '.join(paragraphs)
            
            article_text = self._clean_text(article_text)
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            if self._is_valid_content(article_text):
                return {
                    "success": True,
                    "text": article_text,
                    "method": "beautifulsoup",
                    "error": None,
                    "metadata": {
                        "title": title_text,
                        "url": url,
                        "length": len(article_text)
                    }
                }
            else:
                return {
                    "success": False,
                    "text": None,
                    "method": "beautifulsoup",
                    "error": f"Extracted content too short ({len(article_text)} chars)",
                    "metadata": {}
                }
                
        except requests.Timeout:
            return {"success": False, "text": None, "method": "beautifulsoup", "error": "Request timeout", "metadata": {}}
        except requests.RequestException as e:
            return {"success": False, "text": None, "method": "beautifulsoup", "error": f"Request failed: {str(e)}", "metadata": {}}
        except Exception as e:
            logger.error(f"Scraping error: {str(e)}")
            return {"success": False, "text": None, "method": "beautifulsoup", "error": str(e), "metadata": {}}
    
    async def _playwright_scrape(self, url: str) -> Dict:
        """Playwright browser automation (ASYNC)"""
        if not PLAYWRIGHT_AVAILABLE:
            return {"success": False, "text": None, "method": "playwright", "error": "Playwright not available", "metadata": {}}
        
        try:
            async with async_playwright() as p:
                browser = await p.firefox.launch(
                    headless=True,
                    firefox_user_prefs={
                        "dom.webdriver.enabled": False,
                        "useAutomationExtension": False,
                    }
                )
                
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0',
                    viewport={'width': 1920, 'height': 1080}
                )
                
                page = await context.new_page()
                
                try:
                    await page.goto(url, timeout=30000, wait_until='networkidle')
                except:
                    await page.goto(url, timeout=30000, wait_until='domcontentloaded')
                
                await page.wait_for_timeout(3000)
                
                try:
                    await page.wait_for_selector('article, p, main', timeout=10000)
                except:
                    pass
                
                html = await page.content()
                await browser.close()
            
            soup = BeautifulSoup(html, 'lxml')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'form']):
                element.decompose()
            
            article_text = None
            
            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                article_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            if not article_text or len(article_text) < 200:
                main_content = soup.find('main') or soup.find('div', {'role': 'main'})
                if main_content:
                    paragraphs = main_content.find_all('p')
                    text = ' '.join([p.get_text().strip() for p in paragraphs])
                    if len(text) > len(article_text or ''):
                        article_text = text
            
            if not article_text or len(article_text) < 200:
                paragraphs = soup.find_all('p')
                paragraphs = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                article_text = ' '.join(paragraphs)
            
            article_text = self._clean_text(article_text)
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            if self._is_valid_content(article_text):
                return {
                    "success": True,
                    "text": article_text,
                    "method": "playwright",
                    "error": None,
                    "metadata": {
                        "title": title_text,
                        "url": url,
                        "length": len(article_text)
                    }
                }
            else:
                return {
                    "success": False,
                    "text": None,
                    "method": "playwright",
                    "error": f"Extracted content too short ({len(article_text)} chars)",
                    "metadata": {}
                }
                
        except Exception as e:
            logger.error(f"Playwright scraping error: {str(e)}")
            return {"success": False, "text": None, "method": "playwright", "error": str(e), "metadata": {}}
    
    def _clean_text(self, text: Optional[str]) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def _is_valid_content(self, text: Optional[str]) -> bool:
        """Validate extracted content"""
        if not text or len(text) < 100:
            return False
        alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)
        return alphanumeric_ratio >= 0.5


# Convenience function
async def scrape_url(url: str, timeout: int = 10) -> Dict:
    """Scrape article content from URL (ASYNC)"""
    scraper = URLScraper(timeout=timeout)
    return await scraper.scrape(url)
