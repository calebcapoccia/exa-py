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
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

def to_camel_case(data: dict) -> dict:
    return {
        snake_to_camel(k): to_camel_case(v) if isinstance(v, dict) else v
        for k, v in data.items()
        if v is not None
    }

def camel_to_snake(camel_str: str) -> str:
    snake_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_str).lower()

def to_snake_case(data: dict) -> dict:
    return {
        camel_to_snake(k): to_snake_case(v) if isinstance(v, dict) else v
        for k, v in data.items()
    }

SEARCH_OPTIONS_TYPES = {
    "query": [str],
    "num_results": [int],
    "include_domains": [list],
    "exclude_domains": [list],
    "start_crawl_date": [str],
    "end_crawl_date": [str],
    "start_published_date": [str],
    "end_published_date": [str],
    "include_text": [list],
    "exclude_text": [list],
    "use_autoprompt": [bool],
    "type": [str],
    "category": [str],
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
    "livecrawl_timeout": [int],
    "livecrawl": [LIVECRAWL_OPTIONS],
    "filter_empty_results": [bool],
}

CONTENTS_ENDPOINT_OPTIONS_TYPES = {
    "subpages": [int],
    "subpage_target": [str, list]
}

def validate_search_options(
    options: Dict[str, Optional[object]], expected: dict
) -> None:
    for key, value in options.items():
        if key not in expected:
            raise ValueError(f"Invalid option: '{key}'")
        if value is None:
            continue
        expected_types = expected[key]
        if not any(is_valid_type(value, t) for t in expected_types):
            raise ValueError(
                f"Invalid value for option '{key}': {value}. Expected one of {expected_types}"
            )

def is_valid_type(value, expected_type):
    if get_origin(expected_type) is Literal:
        return value in get_args(expected_type)
    if isinstance(expected_type, type):
        return isinstance(value, expected_type)
    return False

class TextContentsOptions(TypedDict, total=False):
    max_characters: int
    include_html_tags: bool


class HighlightsContentsOptions(TypedDict, total=False):
    query: str
    num_sentences: int
    highlights_per_url: int

class SummaryContentsOptions(TypedDict, total=False):
    query: str

@dataclass
class _Result:
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
    text: Optional[str] = None
    highlights: Optional[List[str]] = None
    highlight_scores: Optional[List[float]] = None
    summary: Optional[str] = None
    subpages: Optional[List[Result]] = None
    extras: Optional[Dict[str, List[str]]] = None

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


@dataclass
class ResultWithText(_Result):
    text: str = dataclasses.field(default_factory=str)

    def __str__(self):
        base_str = super().__str__()
        return base_str + f"Text: {self.text}\n"


@dataclass
class ResultWithHighlights(_Result):
    highlights: List[str] = dataclasses.field(default_factory=list)
    highlight_scores: List[float] = dataclasses.field(default_factory=list)

    def __str__(self):
        base_str = super().__str__()
        return base_str + (
            f"Highlights: {self.highlights}\n"
            f"Highlight Scores: {self.highlight_scores}\n"
        )


@dataclass
class ResultWithTextAndHighlights(_Result):
    text: str = dataclasses.field(default_factory=str)
    highlights: List[str] = dataclasses.field(default_factory=list)
    highlight_scores: List[float] = dataclasses.field(default_factory=list)

    def __str__(self):
        base_str = super().__str__()
        return base_str + (
            f"Text: {self.text}\n"
            f"Highlights: {self.highlights}\n"
            f"Highlight Scores: {self.highlight_scores}\n"
        )

@dataclass
class ResultWithSummary(_Result):
    summary: str = dataclasses.field(default_factory=str)

    def __str__(self):
        base_str = super().__str__()
        return base_str + f"Summary: {self.summary}\n"

@dataclass
class ResultWithTextAndSummary(_Result):
    text: str = dataclasses.field(default_factory=str)
    summary: str = dataclasses.field(default_factory=str)

    def __str__(self):
        base_str = super().__str__()
        return base_str + f"Text: {self.text}\n" + f"Summary: {self.summary}\n"

@dataclass
class ResultWithHighlightsAndSummary(_Result):
    highlights: List[str] = dataclasses.field(default_factory=list)
    highlight_scores: List[float] = dataclasses.field(default_factory=list)
    summary: str = dataclasses.field(default_factory=str)

    def __str__(self):
        base_str = super().__str__()
        return base_str + (
            f"Highlights: {self.highlights}\n"
            f"Highlight Scores: {self.highlight_scores}\n"
            f"Summary: {self.summary}\n"
        )

@dataclass
class ResultWithTextAndHighlightsAndSummary(_Result):
    text: str = dataclasses.field(default_factory=str)
    highlights: List[str] = dataclasses.field(default_factory=list)
    highlight_scores: List[float] = dataclasses.field(default_factory=list)
    summary: str = dataclasses.field(default_factory=str)

    def __str__(self):
        base_str = super().__str__()
        return base_str + (
            f"Text: {self.text}\n"
            f"Highlights: {self.highlights}\n"
            f"Highlight Scores: {self.highlight_scores}\n"
            f"Summary: {self.summary}\n"
        )

T = TypeVar("T")


@dataclass
class SearchResponse(Generic[T]):
    results: List[T]
    autoprompt_string: Optional[str]
    resolved_search_type: Optional[str]
    auto_date: Optional[str]

    def __str__(self):
        output = "\n\n".join(str(result) for result in self.results)
        if self.autoprompt_string:
            output += f"\n\nAutoprompt String: {self.autoprompt_string}"
        if self.resolved_search_type:
            output += f"\nResolved Search Type: {self.resolved_search_type}"

        return output


def nest_fields(original_dict: Dict, fields_to_nest: List[str], new_key: str):
    nested_dict = {}
    for field in fields_to_nest:
        if field in original_dict:
            nested_dict[field] = original_dict.pop(field)
    original_dict[new_key] = nested_dict
    return original_dict


class Exa:
    def __init__(
        self,
        api_key: Optional[str],
        base_url: str = "https://api.exa.ai",
        user_agent: str = "exa-py 1.4.1-beta",
    ):
        if api_key is None:
            import os

            api_key = os.environ.get("EXA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided as argument or in EXA_API_KEY environment variable"
                )
        self.base_url = base_url
        self.headers = {"x-api-key": api_key, "User-Agent": user_agent}

    def request(self, endpoint: str, data):
        res = requests.post(self.base_url + endpoint, json=data, headers=self.headers)
        if res.status_code != 200:
            raise ValueError(
                f"Request failed with status code {res.status_code}: {res.text}"
            )
        return res.json()

    def search(
        self,
        query: str,
        *,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
    ) -> SearchResponse[_Result]:
        options = {k: v for k, v in locals().items() if k != "self" and v is not None}
        validate_search_options(options, SEARCH_OPTIONS_TYPES)
        options = to_camel_case(options)
        data = self.request("/search", options)
        return SearchResponse(
            [Result(**to_snake_case(result)) for result in data["results"]],
            data["autopromptString"] if "autopromptString" in data else None,
            data["resolvedSearchType"] if "resolvedSearchType" in data else None,
            data["autoDate"] if "autoDate" in data else None,
        )

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithText]:
        ...

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithText]:
        ...

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithHighlights]:
        ...

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithTextAndHighlights]:
        ...

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithSummary]:
        ...

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithTextAndSummary]:
        ...

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithHighlightsAndSummary]:
        ...

    @overload
    def search_and_contents(
        self,
        query: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithTextAndHighlightsAndSummary]:
        ...

    def search_and_contents(self, query: str, **kwargs):
        options = {
            k: v
            for k, v in {"query": query, **kwargs}.items()
            if k != "self" and v is not None
        }
        if "text" not in options and "highlights" not in options and "summary" not in options:
            options["text"] = True
        validate_search_options(
            options, {**SEARCH_OPTIONS_TYPES, **CONTENTS_OPTIONS_TYPES}
        )
        options = nest_fields(options, ["text", "highlights", "summary"], "contents")
        options = to_camel_case(options)
        data = self.request("/search", options)
        return SearchResponse(
            [Result(**to_snake_case(result)) for result in data["results"]],
            data["autopromptString"] if "autopromptString" in data else None,
            data["autoDate"] if "autoDate" in data else None,
            data["resolvedSearchType"] if "resolvedSearchType" in data else None,
        )

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithText]:
        ...

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        *,
        text: Union[TextContentsOptions, Literal[True]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithText]:
        ...

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        *,
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithHighlights]:
        ...

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        *,
        text: Union[TextContentsOptions, Literal[True]],
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithTextAndHighlights]:
        ...

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        *,
        summary: Union[SummaryContentsOptions, Literal[True]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithSummary]:
        ...

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        *,
        text: Union[TextContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithTextAndSummary]:
        ...

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        *,
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithHighlightsAndSummary]:
        ...

    @overload
    def get_contents(
        self,
        ids: Union[str, List[str], List[_Result]],
        *,
        text: Union[TextContentsOptions, Literal[True]],
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[Union[str, List[str]]] = None
    ) -> SearchResponse[ResultWithTextAndHighlightsAndSummary]:
        ...

    def get_contents(self, ids: Union[str, List[str], List[_Result]], **kwargs):
        options = {
            k: v
            for k, v in {"ids": ids, **kwargs}.items()
            if k != "self" and v is not None
        }
        if "text" not in options and "highlights" not in options and "summary" not in options:
            options["text"] = True
        validate_search_options(options, {**CONTENTS_OPTIONS_TYPES, **CONTENTS_ENDPOINT_OPTIONS_TYPES})
        options = to_camel_case(options)
        data = self.request("/contents", options)
        return SearchResponse(
            [Result(**to_snake_case(result)) for result in data["results"]],
            data["autopromptString"] if "autopromptString" in data else None,
            data["resolvedSearchType"] if "resolvedSearchType" in data else None,
            data["autoDate"] if "autoDate" in data else None,
        )

    def find_similar(
        self,
        url: str,
        *,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
    ) -> SearchResponse[_Result]:
        options = {k: v for k, v in locals().items() if k != "self" and v is not None}
        validate_search_options(options, FIND_SIMILAR_OPTIONS_TYPES)
        options = to_camel_case(options)
        data = self.request("/findSimilar", options)
        return SearchResponse(
            [Result(**to_snake_case(result)) for result in data["results"]],
            data["autopromptString"] if "autopromptString" in data else None,
            data["resolvedSearchType"] if "resolvedSearchType" in data else None,
            data["autoDate"] if "autoDate" in data else None,
        )

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithText]:
        ...

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithText]:
        ...

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithHighlights]:
        ...

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithTextAndHighlights]:
        ...

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithSummary]:
        ...

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithTextAndSummary]:
        ...

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithHighlightsAndSummary]:
        ...

    @overload
    def find_similar_and_contents(
        self,
        url: str,
        *,
        text: Union[TextContentsOptions, Literal[True]],
        highlights: Union[HighlightsContentsOptions, Literal[True]],
        summary: Union[SummaryContentsOptions, Literal[True]],
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        exclude_source_domain: Optional[bool] = None,
        category: Optional[str] = None,
        livecrawl_timeout: Optional[int] = None,
        livecrawl: Optional[LIVECRAWL_OPTIONS] = None,
        filter_empty_results: Optional[bool] = None,
    ) -> SearchResponse[ResultWithTextAndHighlightsAndSummary]:
        ...

    def find_similar_and_contents(self, url: str, **kwargs):
        options = {
            k: v
            for k, v in {"url": url, **kwargs}.items()
            if k != "self" and v is not None
        }
        if "text" not in options and "highlights" not in options:
            options["text"] = True
        validate_search_options(
            options, {**FIND_SIMILAR_OPTIONS_TYPES, **CONTENTS_OPTIONS_TYPES}
        )
        options = to_camel_case(options)
        options = nest_fields(options, ["text", "highlights", "summary"], "contents")
        data = self.request("/findSimilar", options)
        return SearchResponse(
            [Result(**to_snake_case(result)) for result in data["results"]],
            data["autopromptString"] if "autopromptString" in data else None,
            data["resolvedSearchType"] if "resolvedSearchType" in data else None,
            data["autoDate"] if "autoDate" in data else None,
        )

    def wrap(self, client: OpenAI):
        func = client.chat.completions.create

        @wraps(func)
        def create_with_rag(
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            use_exa: Optional[Literal["required", "none", "auto"]] = "auto",
            highlights: Union[HighlightsContentsOptions, Literal[True], None] = None,
            num_results: Optional[int] = 3,
            include_domains: Optional[List[str]] = None,
            exclude_domains: Optional[List[str]] = None,
            start_crawl_date: Optional[str] = None,
            end_crawl_date: Optional[str] = None,
            start_published_date: Optional[str] = None,
            end_published_date: Optional[str] = None,
            include_text: Optional[List[str]] = None,
            exclude_text: Optional[List[str]] = None,
            use_autoprompt: Optional[bool] = True,
            type: Optional[str] = None,
            category: Optional[str] = None,
            result_max_len: int = 2048,
            **openai_kwargs,
        ):
            exa_kwargs = {
                "num_results": num_results,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "highlights": highlights,
                "start_crawl_date": start_crawl_date,
                "end_crawl_date": end_crawl_date,
                "start_published_date": start_published_date,
                "end_published_date": end_published_date,
                "include_text": include_text,
                "exclude_text": exclude_text,
                "use_autoprompt": use_autoprompt,
                "type": type,
                "category": category,
            }

            create_kwargs = {
                "model": model,
                **openai_kwargs,
            }

            return self._create_with_tool(
                create_fn=func,
                messages=list(messages),
                max_len=result_max_len,
                create_kwargs=create_kwargs,
                exa_kwargs=exa_kwargs,
            )

        print("Wrapping OpenAI client with Exa functionality.", type(create_with_rag))
        client.chat.completions.create = create_with_rag # type: ignore

        return client

    def _create_with_tool(
        self,
        create_fn: Callable,
        messages: List[ChatCompletionMessageParam],
        max_len,
        create_kwargs,
        exa_kwargs,
    ) -> ExaOpenAICompletion:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for relevant information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        create_kwargs["tools"] = tools

        completion = create_fn(messages=messages, **create_kwargs)

        query = maybe_get_query(completion)

        if not query:
            return ExaOpenAICompletion.from_completion(completion=completion, exa_result=None)

        exa_result = self.search_and_contents(query, **exa_kwargs)
        exa_str = format_exa_result(exa_result, max_len=max_len)
        new_messages = add_message_to_messages(completion, messages, exa_str)
        completion = create_fn(messages=new_messages, **create_kwargs)

        exa_completion = ExaOpenAICompletion.from_completion(
            completion=completion, exa_result=exa_result
        )
        return exa_completion