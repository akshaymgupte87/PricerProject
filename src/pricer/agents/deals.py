from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Self, Literal
from bs4 import BeautifulSoup
import ipaddress
import logging
import re
import feedparser
from tqdm import tqdm
import requests
import time
from urllib.parse import urljoin, urlparse

from pricer.agents.guardrails import validate_price


def validate_public_http_url(value: str) -> str:
    """Reject malformed URLs and obvious local/private targets before fetching."""
    parsed = urlparse(value)
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("URL has an invalid host or port") from error
    if (parsed.scheme not in {"http", "https"} or not hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("URL must be an absolute public HTTP(S) URL")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("URL port must be between 1 and 65535")
    host = hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("URL host must not be local")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("URL host must be a public IP address")
    return value


def get_public_url(url: str, *, timeout: tuple[int, int] = (5, 15)):
    """Fetch a URL while validating every redirect destination before requesting it."""
    current = url
    for redirect_count in range(6):
        validate_public_http_url(current)
        response = requests.get(current, timeout=timeout, allow_redirects=False)
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location")
            if not location:
                raise ValueError("redirect response did not include a Location")
            if redirect_count == 5:
                raise requests.TooManyRedirects("too many redirects while fetching public URL")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        return response
    raise requests.TooManyRedirects("too many redirects while fetching public URL")

feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

# You could also add: "https://www.dealnews.com/c238/Automotive/?rss=1"
# "https://www.dealnews.com/c196/Home-Garden/?rss=1"


def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = soup.get_text(" ", strip=True)
    return result.replace("\n", " ")


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.url = entry["links"][0]["href"]
        response = get_public_url(self.url)
        stuff = response.content
        soup = BeautifulSoup(stuff, "html.parser")
        content_section = soup.find("div", class_="content-section")
        # Deal sites change their markup. Keep the RSS summary usable when the
        # detail container disappears instead of crashing the whole feed scan.
        content = content_section.get_text() if content_section else self.summary
        content = content.replace("\nmore", "").replace("\n", " ")
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
        self.truncate()

    def truncate(self):
        """
        Limit the fields to a sensible length to avoid sending too much info to the model
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            try:
                response = get_public_url(feed_url)
                feed = feedparser.parse(response.content)
            except Exception:
                logging.warning("Could not retrieve RSS feed %s", feed_url, exc_info=True)
                continue
            for entry in feed.entries[:10]:
                try:
                    deals.append(cls(entry))
                except Exception:
                    logging.warning("Skipping unreadable deal from %s", feed_url, exc_info=True)
                else:
                    time.sleep(0.05)
        return deals


class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")
    category: str | None = Field(default=None, description="Optional product category")
    brand: str | None = Field(default=None, description="Optional product brand")
    condition: str | None = Field(default=None, description="New, used, or refurbished")
    image_url: str | None = Field(default=None, description="Optional product image URL")

    @field_validator("product_description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("product description cannot be empty")
        return cleaned[:2_000]

    @field_validator("price")
    @classmethod
    def validate_deal_price(cls, value: float) -> float:
        return validate_price(value)

    @field_validator("url")
    @classmethod
    def validate_deal_url(cls, value: str) -> str:
        return validate_public_http_url(value)


class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """

    deal: Deal
    estimate: float
    discount: float
    model_estimates: Dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_low: float | None = None
    confidence_high: float | None = None
    evidence: List[Dict] = Field(default_factory=list)
    explanation: str = ""
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: str | None = None
