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
    "include_text": [
        list
    ],  # list of strings that must be present in webpage text of results. Currently, only one string is supported, up to 5 words.
    "exclude_text": [list],  # list of strings that must not be present in webpage text of result. Currently, only one string is supported, up to 5 words.
    "use_autoprompt": [bool],  # Convert query to Exa (Higher latency, Default: false).
    "type": [
        str
    ],  # 'keyword' or 'neural' (Default: neural). Choose 'neural' for high-quality, semantically relevant content in popular domains. 'Keyword' is for specific, local, or obscure queries.
    "category": [
        str
    ],  # A data category to focus on, with higher comprehensivity and data cleanliness.
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

# the livecrawl options
LIVECRAWL_OPTIONS = Literal["always", "fallback", "never"]

CONTENTS_OPTIONS_TYPES = {
    "ids": [list],
    "text": [dict, bool],
    "highlights": [dict, bool],
    "summary": [dict, bool],
    "livecrawl_timeout": [int],
    "livecrawl": [LIVECRAWL_OPTIONS],
    "filter_empty_results": [bool],
    "subpages": [int],  # Number of subpages to get contents for
    "subpage_target": [str, list],  # Specific subpage(s) to get contents for
    "extras": [dict],  # Extra parameters to pass
}

class TextContentsOptions(TypedDict, total=False):
    """A class representing the options that you can specify when requesting text

    Attributes:
        max_characters (int): The maximum number of characters to return. Default: None (no limit).
        include_html_tags (bool): If true, include HTML tags in the returned text. Default false.
    """

    max_characters: int
    include_html_tags: bool


class HighlightsContentsOptions(TypedDict, total=False):
    """A class representing the options that you can specify when requesting highlights

    Attributes:
        query (str): The query string for the highlights. if not specified, defaults to a generic summarization query.
        num_sentences (int): Size of highlights to return, in sentences. Default: 5
        highlights_per_url (int): The number of highlights to return per URL. Default: 1
    """

    query: str
    num_sentences: int
    highlights_per_url: int

class SummaryContentsOptions(TypedDict, total=False):
    """A class representing the options that you can specify when requesting summary

    Attributes:
        query (str): The query string for the summary. Summary will bias towards answering the query.
    """

    query: str

class ExtrasContentsOptions(TypedDict, total=False):
    """A class representing the options that you can specify when requesting extras

    Attributes:
        links (int): The number of links to return.
    """

    links: int

@dataclass
class _Result:
    """A class representing the base fields of a search result.

    Attributes:
        title (str): The title of the search result.
        url (str): The URL of the search result.
        id (str): The temporary ID for the document.
        score (float, optional): A number from 0 to 1 representing similarity between the query/url and the result.
        published_date (str, optional): An estimate of the creation date, from parsing HTML content.
        author (str, optional): If available, the author of the content.
        image (str, optional): The URL of an image associated with the search result.
    """

    url: str
    id: str
    title: Optional[str] = None
    score: Optional[float] = None
    published_date: Optional[str] = None
    author: Optional[str] = None
    image: Optional[str] = None

    def __init__(self, **kwargs):
        self.url = kwargs['url']
        self.id = kwargs['id']
        self.title = kwargs.get('title')
        self.score = kwargs.get('score')
        self.published_date = kwargs.get('published_date')
        self.author = kwargs.get('author')
        self.image = kwargs.get('image')

    def __str__(self):
        return (
            f"Title: {self.title}\n"
            f"URL: {self.url}\n"
            f"ID: {self.id}\n"
            f"Score: {self.score}\n"
            f"Published Date: {self.published_date}\n"
            f"Author: {self.author}\n"
            f"Image: {self.image}\n"
        )


@dataclass
class Result(_Result):
    """
    A class representing a search result with optional text, highlights, summary, subpages and extras.

    Attributes:
        text (str, optional): The text of the search result page.
        highlights (List[str], optional): The highlights of the search result.
        highlight_scores (List[float], optional): The scores of the highlights of the search result.
        summary (str, optional): The summary of the search result.
        subpages (List[Result], optional): The subpages of the search result.
        extras (Dict, optional): Extra information like links from the search result.
    """

    text: Optional[str] = None
    highlights: Optional[List[str]] = None
    highlight_scores: Optional[List[float]] = None
    summary: Optional[str] = None
    subpages: Optional[List[Result]] = dataclasses.field(default_factory=list)
    extras: Optional[Dict[str, Union[str, List[str]]]] = dataclasses.field(default_factory=dict)

    def __str__(self):
        base_str = super().__str__()
        return base_str + (
            f"Text: {self.text}\n"
            f"Highlights: {self.highlights}\n"
            f"Highlight Scores: {self.highlight_scores}\n"
            f"Summary: {self.summary}\n"
            f"Subpages: {self.subpages}\n"
            f"Extras: {self.extras}\n"
        )

[Rest of the SDK classes and methods remain the same as they follow the same pattern]