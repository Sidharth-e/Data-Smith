"""
Browser Automation Service for Data Smith.
Provides web scraping and browser automation using Playwright.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# Playwright import with fallback
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from models.scrape_config import (
    ScrapeConfig, ScrapeResult, ScrapedItem,
    SelectorConfig, ActionConfig, ExtractionType, WaitCondition
)


class BrowserServiceError(Exception):
    """Exception raised by the browser service."""
    pass


class BrowserAutomationService:
    """
    Browser automation service using Playwright.
    
    Features:
    - Configurable scraping workflows
    - JavaScript rendering support
    - Multiple extraction types (text, tables, links, images)
    - Pagination handling
    - Action sequences (click, type, scroll)
    - Screenshot capture
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the browser service.
        
        Args:
            headless: Run browser in headless mode
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise BrowserServiceError(
                "Playwright not available. Install with: pip install playwright && playwright install"
            )
        
        self.headless = headless
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self):
        """Start the browser."""
        if self._browser:
            return
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context()
    
    async def close(self):
        """Close the browser."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if hasattr(self, '_playwright'):
            await self._playwright.stop()
    
    async def scrape(self, config: ScrapeConfig) -> ScrapeResult:
        """
        Execute a scraping job based on configuration.
        
        Args:
            config: Scrape configuration
            
        Returns:
            ScrapeResult with extracted data
        """
        start_time = datetime.now()
        errors = []
        items = []
        pages_scraped = 0
        
        try:
            await self.start()
            
            # Get URLs to scrape
            urls = config.urls or [config.url]
            
            for url in urls:
                try:
                    page_items = await self._scrape_url(url, config)
                    items.extend(page_items)
                    pages_scraped += 1
                    
                    # Delay between requests
                    if config.delay_between_requests > 0 and url != urls[-1]:
                        await asyncio.sleep(config.delay_between_requests / 1000)
                        
                except Exception as e:
                    errors.append(f"Error scraping {url}: {str(e)}")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return ScrapeResult(
                success=len(items) > 0,
                items=items,
                total_items=len(items),
                pages_scraped=pages_scraped,
                errors=errors,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return ScrapeResult(
                success=False,
                items=[],
                total_items=0,
                pages_scraped=pages_scraped,
                errors=[str(e)],
                duration_seconds=duration
            )
    
    async def _scrape_url(
        self, 
        url: str, 
        config: ScrapeConfig
    ) -> List[ScrapedItem]:
        """Scrape a single URL."""
        page = await self._context.new_page()
        items = []
        
        try:
            # Set viewport
            await page.set_viewport_size({
                "width": config.viewport_width,
                "height": config.viewport_height
            })
            
            # Set user agent if specified
            if config.user_agent:
                await page.set_extra_http_headers({"User-Agent": config.user_agent})
            
            # Navigate to URL
            await page.goto(url, timeout=config.timeout)
            
            # Wait for condition
            await self._wait_for_condition(page, config)
            
            # Execute actions before extraction
            for action in config.actions:
                await self._execute_action(page, action)
            
            # Handle pagination if enabled
            if config.pagination and config.pagination.enabled:
                items = await self._scrape_with_pagination(page, url, config)
            else:
                # Single page extraction
                data = await self._extract_data(page, config.selectors)
                page_title = await page.title()
                
                items.append(ScrapedItem(
                    url=url,
                    data=data,
                    timestamp=datetime.now().isoformat(),
                    page_title=page_title,
                    metadata={"source": "single_page"}
                ))
            
        finally:
            await page.close()
        
        return items
    
    async def _wait_for_condition(self, page: Page, config: ScrapeConfig):
        """Wait for the specified condition."""
        if config.wait_for == WaitCondition.SELECTOR and config.wait_for_selector:
            await page.wait_for_selector(
                config.wait_for_selector, 
                timeout=config.wait_timeout
            )
        elif config.wait_for == WaitCondition.NETWORKIDLE:
            await page.wait_for_load_state("networkidle")
        elif config.wait_for == WaitCondition.LOAD:
            await page.wait_for_load_state("load")
        elif config.wait_for == WaitCondition.DOMCONTENTLOADED:
            await page.wait_for_load_state("domcontentloaded")
    
    async def _execute_action(self, page: Page, action: ActionConfig):
        """Execute a browser action."""
        if action.action == "click" and action.selector:
            await page.click(action.selector)
        elif action.action == "type" and action.selector and action.value:
            await page.fill(action.selector, action.value)
        elif action.action == "scroll":
            if action.selector:
                await page.evaluate(f'document.querySelector("{action.selector}").scrollIntoView()')
            else:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif action.action == "wait":
            wait_time = int(action.value) if action.value else 1000
            await asyncio.sleep(wait_time / 1000)
        elif action.action == "navigate" and action.value:
            await page.goto(action.value)
        
        # Wait after action
        if action.wait_after > 0:
            await asyncio.sleep(action.wait_after / 1000)
    
    async def _extract_data(
        self, 
        page: Page, 
        selectors: List[SelectorConfig]
    ) -> Dict[str, Any]:
        """Extract data based on selectors."""
        data = {}
        
        for selector in selectors:
            try:
                value = await self._extract_selector(page, selector)
                data[selector.name] = value
            except Exception as e:
                if selector.required:
                    raise BrowserServiceError(f"Required field '{selector.name}' not found: {e}")
                data[selector.name] = selector.default
        
        return data
    
    async def _extract_selector(
        self, 
        page: Page, 
        selector: SelectorConfig
    ) -> Any:
        """Extract data for a single selector."""
        # Get elements
        if selector.multiple:
            elements = await page.query_selector_all(selector.selector)
        else:
            element = await page.query_selector(selector.selector)
            elements = [element] if element else []
        
        if not elements:
            return selector.default
        
        # Extract based on type
        results = []
        for element in elements:
            if selector.extraction_type == ExtractionType.TEXT:
                value = await element.text_content()
            elif selector.extraction_type == ExtractionType.HTML:
                value = await element.inner_html()
            elif selector.extraction_type == ExtractionType.ATTRIBUTE:
                value = await element.get_attribute(selector.attribute or "href")
            elif selector.extraction_type == ExtractionType.LINK:
                value = {
                    "text": await element.text_content(),
                    "href": await element.get_attribute("href")
                }
            elif selector.extraction_type == ExtractionType.IMAGE:
                value = {
                    "src": await element.get_attribute("src"),
                    "alt": await element.get_attribute("alt")
                }
            elif selector.extraction_type == ExtractionType.TABLE:
                value = await self._extract_table(element)
            else:
                value = await element.text_content()
            
            results.append(value.strip() if isinstance(value, str) else value)
        
        return results if selector.multiple else (results[0] if results else selector.default)
    
    async def _extract_table(self, element) -> List[Dict[str, str]]:
        """Extract table data from a table element."""
        rows = await element.query_selector_all("tr")
        if not rows:
            return []
        
        # Get headers
        header_row = rows[0]
        header_cells = await header_row.query_selector_all("th, td")
        headers = [await cell.text_content() for cell in header_cells]
        headers = [h.strip() for h in headers]
        
        # Get data rows
        table_data = []
        for row in rows[1:]:
            cells = await row.query_selector_all("td")
            if cells:
                row_data = {}
                for i, cell in enumerate(cells):
                    key = headers[i] if i < len(headers) else f"column_{i}"
                    row_data[key] = (await cell.text_content()).strip()
                table_data.append(row_data)
        
        return table_data
    
    async def _scrape_with_pagination(
        self, 
        page: Page, 
        url: str, 
        config: ScrapeConfig
    ) -> List[ScrapedItem]:
        """Scrape with pagination support."""
        items = []
        pagination = config.pagination
        page_num = 1
        
        while page_num <= pagination.max_pages:
            # Extract current page
            data = await self._extract_data(page, config.selectors)
            page_title = await page.title()
            
            items.append(ScrapedItem(
                url=page.url,
                data=data,
                timestamp=datetime.now().isoformat(),
                page_title=page_title,
                metadata={"page_number": page_num}
            ))
            
            # Check for empty results
            if pagination.stop_on_empty and not any(data.values()):
                break
            
            # Try to go to next page
            if pagination.next_button_selector:
                try:
                    next_button = await page.query_selector(pagination.next_button_selector)
                    if not next_button:
                        break
                    
                    await next_button.click()
                    await asyncio.sleep(pagination.wait_between_pages / 1000)
                    await self._wait_for_condition(page, config)
                    page_num += 1
                except Exception:
                    break
            else:
                break
        
        return items
    
    async def take_screenshot(
        self, 
        url: str, 
        full_page: bool = True
    ) -> bytes:
        """Take a screenshot of a URL."""
        await self.start()
        page = await self._context.new_page()
        
        try:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            return await page.screenshot(full_page=full_page, type="png")
        finally:
            await page.close()
    
    async def get_page_content(self, url: str) -> Dict[str, Any]:
        """Get raw page content and metadata."""
        await self.start()
        page = await self._context.new_page()
        
        try:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            
            return {
                "url": url,
                "title": await page.title(),
                "html": await page.content(),
                "text": await page.evaluate("() => document.body.innerText"),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            await page.close()


# Convenience function
async def scrape_url(url: str, selectors: List[Dict] = None) -> ScrapeResult:
    """
    Quick function to scrape a URL.
    
    Args:
        url: URL to scrape
        selectors: List of selector configs
        
    Returns:
        ScrapeResult with extracted data
    """
    config = ScrapeConfig(
        url=url,
        selectors=[SelectorConfig(**s) for s in (selectors or [])]
    )
    
    async with BrowserAutomationService() as service:
        return await service.scrape(config)
