from __future__ import annotations
from dataclasses import dataclass
import dataclasses
from functools import wraps
import re
import requests
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Dict,
    Generic,
    TypeVar,
    overload,
    Union,
    Literal,
    get_origin,
    get_args
)
from typing_extensions import TypedDict

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat_model import ChatModel
from exa_py.utils import (
    ExaOpenAICompletion,
    add_message_to_messages,
    format_exa_result,
    maybe_get_query,
)
import os

is_beta = os.getenv("IS_BETA") == "True"

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case string to camelCase.

    Args:
        snake_str (str): The string in snake_case format.

    Returns:
        str: The string converted to camelCase format.
    """
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

def to_camel_case(data: dict) -> dict:
    """
    Convert keys in a dictionary from snake_case to camelCase recursively.

    Args:
        data (dict): The dictionary with keys in snake_case format.

    Returns:
        dict: The dictionary with keys converted to camelCase format.
    """
    return {
        snake_to_camel(k): to_camel_case(v) if isinstance(v, dict) else v
        for k, v in data.items()
        if v is not None
    }

def camel_to_snake(camel_str: str) -> str:
    """Convert camelCase string to snake_case.

    Args:
        camel_str (str): The string in camelCase format.

    Returns:
        str: The string converted to snake_case format.
    """
    snake_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_str).lower()

def to_snake_case(data: dict) -> dict:
    """
    Convert keys in a dictionary from camelCase to snake_case recursively.

    Args:
        data (dict): The dictionary with keys in camelCase format.

    Returns:
        dict: The dictionary with keys converted to snake_case format.
    """
    return {
        camel_to_snake(k): to_snake_case(v) if isinstance(v, dict) else v
        for k, v in data.items()
    }

SEARCH_OPTIONS_TYPES = {
    "query": [str],  # The query string.
    "num_results": [int],  # Number of results (Default: 10, Max for basic: 10).
    "include_domains": [
        list
    ],  # Domains to search from; exclusive with 'exclude_domains'.
    "exclude_domains": [list],  # Domains to omit; exclusive with 'include_domains'.
    "start_crawl_date": [str],  # Results after this crawl date. ISO 8601 format.
    "end_crawl_date": [str],  # Results before this crawl date. ISO 8601 format.
    "start_published_date": [
        str
    ],  # Results after this publish date; excludes links with no date. ISO 8601 format.
    "end_published_date": [
        str
    ],  # Results before this publish date; excludes links with no date. ISO 8601 format.
    "include_text": [list],  # list of strings that must be present in webpage text of results.
    "exclude_text": [list],  # list of strings that must not be present in webpage text of result.
    "use_autoprompt": [bool],  # Convert query to Exa (Higher latency, Default: false).
    "type": [str],  # 'keyword' or 'neural' (Default: neural).
    "category": [str],  # A data category to focus on.
}

FIND_SIMILAR_OPTIONS_TYPES = {
    "url": [str],
    "num_results": [int],
    "include_domains": [list],
    "exclude_domains": [list],
    "start_crawl_date": [str],
    "end_crawl_date": [str],
    "start_published_date": [str],
    "end_published_date": [str],
    "include_text": [list],
    "exclude_text": [list],
    "exclude_source_domain": [bool],
    "category": [str],
}

LIVECRAWL_OPTIONS = Literal["always", "fallback", "never"]

CONTENTS_OPTIONS_TYPES = {
    "ids": [list],
    "text": [dict, bool],
    "highlights": [dict, bool],
    "summary": [dict, bool],
    "metadata": [dict, bool],
    "livecrawl_timeout": [int],
    "livecrawl": [LIVECRAWL_OPTIONS],
    "filter_empty_results": [bool],
    "subpages": [int],
    "subpage_target": [str, list],
    "extras": [dict],
}

class TextContentsOptions(TypedDict, total=False):
    """A class representing the options for text content retrieval

    Attributes:
        max_characters (int): The maximum number of characters to return.
        include_html_tags (bool): If true, include HTML tags in the returned text.
    """
    max_characters: int
    include_html_tags: bool

class HighlightsContentsOptions(TypedDict, total=False):
    """A class representing the options for highlights retrieval

    Attributes:
        query (str): The query string for the highlights.
        num_sentences (int): Size of highlights in sentences.
        highlights_per_url (int): Number of highlights per URL.
    """
    query: str
    num_sentences: int
    highlights_per_url: int

class SummaryContentsOptions(TypedDict, total=False):
    """A class representing the options for summary retrieval

    Attributes:
        query (str): The query string for the summary.
    """
    query: str

class ExtrasOptions(TypedDict, total=False):
    """A class representing the extra options for content retrieval

    Attributes:
        links (int): The number of links to return.
    """
    links: int

@dataclass
class _Result:
    """A class representing a search result.

    Attributes:
        title (str): The title of the search result.
        url (str): The URL of the search result.
        id (str): The temporary ID for the document.
        score (float, optional): Similarity score between query/url and result.
        published_date (str, optional): Estimated creation date.
        author (str, optional): Content author if available.
        image (str, optional): Associated image URL if available.
    """
    url: str
    id: str
    title: Optional[str] = None
    score: Optional[float] = None
    published_date: Optional[str] = None
    author: Optional[str] = None
    image: Optional[str] = None
    subpages: Optional[List[Result]] = None
    extras: Optional[Dict] = None

    def __init__(self, **kwargs):
        self.url = kwargs['url']
        self.id = kwargs['id']
        self.title = kwargs.get('title')
        self.score = kwargs.get('score')
        self.published_date = kwargs.get('published_date')
        self.author = kwargs.get('author')
        self.image = kwargs.get('image')
        self.subpages = kwargs.get('subpages')
        self.extras = kwargs.get('extras')

    def __str__(self):
        base_str = (
            f"Title: {self.title}\n"
            f"URL: {self.url}\n"
            f"ID: {self.id}\n"
            f"Score: {self.score}\n"
            f"Published Date: {self.published_date}\n"
            f"Author: {self.author}\n"
            f"Image: {self.image}\n"
        )
        if self.subpages:
            base_str += f"Subpages: {len(self.subpages)} pages\n"
        if self.extras:
            base_str += f"Extras: {self.extras}\n"
        return base_str

[Previous classes and methods continue as before with updated Result classes to include subpages and extras]