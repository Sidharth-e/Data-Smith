"""
Scrape Configuration Models for Data Smith.
Defines configuration for browser automation and web scraping.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum


class SelectorType(str, Enum):
    """Type of CSS/XPath selector."""
    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"
    ID = "id"
    CLASS = "class"


class WaitCondition(str, Enum):
    """Conditions to wait for before extraction."""
    LOAD = "load"
    DOMCONTENTLOADED = "domcontentloaded"
    NETWORKIDLE = "networkidle"
    SELECTOR = "selector"


class ExtractionType(str, Enum):
    """Type of data to extract from elements."""
    TEXT = "text"
    HTML = "html"
    ATTRIBUTE = "attribute"
    TABLE = "table"
    LINK = "link"
    IMAGE = "image"


class SelectorConfig(BaseModel):
    """Configuration for a single selector."""
    name: str = Field(..., description="Name for the extracted field")
    selector: str = Field(..., description="CSS selector or XPath expression")
    selector_type: SelectorType = Field(default=SelectorType.CSS)
    extraction_type: ExtractionType = Field(default=ExtractionType.TEXT)
    attribute: Optional[str] = Field(default=None, description="Attribute to extract if extraction_type is 'attribute'")
    multiple: bool = Field(default=False, description="Whether to extract all matching elements")
    required: bool = Field(default=False, description="Whether this field is required")
    default: Optional[str] = Field(default=None, description="Default value if not found")


class ActionConfig(BaseModel):
    """Configuration for a browser action."""
    action: Literal["click", "type", "scroll", "wait", "screenshot", "navigate"]
    selector: Optional[str] = None
    value: Optional[str] = None
    wait_after: int = Field(default=500, description="Milliseconds to wait after action")


class PaginationConfig(BaseModel):
    """Configuration for pagination handling."""
    enabled: bool = Field(default=False)
    next_button_selector: Optional[str] = None
    max_pages: int = Field(default=10)
    wait_between_pages: int = Field(default=1000, description="Milliseconds to wait between pages")
    stop_on_empty: bool = Field(default=True)


class ScrapeConfig(BaseModel):
    """Complete configuration for a scraping job."""
    # Target
    url: str = Field(..., description="URL to scrape")
    urls: Optional[List[str]] = Field(default=None, description="Multiple URLs to scrape")
    
    # Browser settings
    headless: bool = Field(default=True, description="Run browser in headless mode")
    javascript_enabled: bool = Field(default=True)
    timeout: int = Field(default=30000, description="Page load timeout in milliseconds")
    viewport_width: int = Field(default=1920)
    viewport_height: int = Field(default=1080)
    user_agent: Optional[str] = None
    
    # Wait conditions
    wait_for: WaitCondition = Field(default=WaitCondition.NETWORKIDLE)
    wait_for_selector: Optional[str] = None
    wait_timeout: int = Field(default=10000)
    
    # Extraction
    selectors: List[SelectorConfig] = Field(default_factory=list)
    
    # Actions before extraction
    actions: List[ActionConfig] = Field(default_factory=list)
    
    # Pagination
    pagination: Optional[PaginationConfig] = None
    
    # Output
    output_format: Literal["json", "csv", "raw"] = Field(default="json")
    include_metadata: bool = Field(default=True)
    
    # Rate limiting
    delay_between_requests: int = Field(default=1000, description="Milliseconds between requests")


class ScrapedItem(BaseModel):
    """A single scraped data item."""
    url: str
    data: Dict[str, Any]
    timestamp: str
    page_title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScrapeResult(BaseModel):
    """Result of a scraping operation."""
    success: bool
    items: List[ScrapedItem] = Field(default_factory=list)
    total_items: int = 0
    pages_scraped: int = 1
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class ScrapeJobStatus(BaseModel):
    """Status of a scrape job."""
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: float = 0.0
    message: str = ""
    result: Optional[ScrapeResult] = None
