import json
import logging
import os
from typing import Callable, List, Union

import dspy
import requests
from openai import OpenAI

from utils import WebPageHelper


class YouRM(dspy.Retrieve):
    def __init__(self, ydc_api_key=None, k=3, is_valid_source: Callable = None):
        super().__init__(k=k)
        if not ydc_api_key and not os.environ.get("YDC_API_KEY"):
            raise RuntimeError(
                "You must supply ydc_api_key or set environment variable YDC_API_KEY"
            )
        elif ydc_api_key:
            self.ydc_api_key = ydc_api_key
        else:
            self.ydc_api_key = os.environ["YDC_API_KEY"]
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"YouRM": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with You.com for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        for query in queries:
            try:
                headers = {"X-API-Key": self.ydc_api_key}
                results = requests.get(
                    f"https://api.ydc-index.io/search?query={query}",
                    headers=headers,
                ).json()

                authoritative_results = []
                for r in results["hits"]:
                    if self.is_valid_source(r["url"]) and r["url"] not in exclude_urls:
                        authoritative_results.append(r)
                if "hits" in results:
                    collected_results.extend(authoritative_results[: self.k])
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        print(f"{collected_results=}")
        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(
        self,
        bing_search_api_key=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        mkt="en-US",
        language="en",
        **kwargs,
    ):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY"
            )
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {"mkt": mkt, "setLang": language, "count": k, **kwargs}
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"BingSearch": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint, headers=headers, params={**self.params, "q": query}
                ).json()

                for d in results["webPages"]["value"]:
                    if self.is_valid_source(d["url"]) and d["url"] not in exclude_urls:
                        url_to_results[d["url"]] = {
                            "url": d["url"],
                            "title": d["name"],
                            "description": d["snippet"],
                        }
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(
            list(url_to_results.keys())
        )
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r["snippets"] = valid_url_to_snippets[url]["snippets"]
            collected_results.append(r)

        return collected_results


class Chat:
    def __init__(self, openai_api_key=None, model="gpt-4o"):
        """
        Initializes the Chat class with API key and model.
        """
        api_key = None
        if not openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "You must supply openai_api_key or set environment variable OPENAI_API_KEY"
            )
        elif openai_api_key:
            api_key = openai_api_key
        else:
            api_key = os.environ["OPENAI_API_KEY"]

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            }
        ]

    def append_response_to_history(self, response):
        message = response.choices[0].message

        self.chat_history.append({"role": message.role, "content": message.content})

    def get_reply(self, response):
        content = response.choices[0].message.content
        return content

    def print_reply(self, response):
        print(self.get_reply(response))

    def ask(self, query, model="gpt-4o"):
        self.chat_history.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=self.chat_history,
        )

        self.append_response_to_history(response)

        # self.print_reply(response)
        # print(response.usage.total_tokens)
        return self.get_reply(response)

    def browser_results(self, query, k=3):
        _query = f"Use Browser to retrieve {k} results related to {query}. The output should be a results array of dict of (title, url, description, snippets(array of string))"
        json_content = self.ask(_query)
        return json.loads(json_content)


class OpenAIBrowserSearch(dspy.Retrieve):
    model = "gpt-4o"

    def __init__(
        self,
        openai_api_key=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        **kwargs,
    ):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            **kwargs: Additional parameters for the OpenAI API.
        """
        super().__init__(k=k)
        if not openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "You must supply openai_api_key or set environment variable OPENAI_API_KEY"
            )
        elif openai_api_key:
            self.openai_api_key = openai_api_key
        else:
            self.openai_api_key = os.environ["OPENAI_API_KEY"]

        self.k = k
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True
        self.chat = Chat(openai_api_key=self.openai_api_key, model=self.model)

    # TODO: this is used in BingSearch and YouRM, consider moving it to a common place.
    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"OpenAIBrowserSearch": usage}

    # this is the important function that is used by Retriever in the pipeline
    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with OpenAI's Browser Tool for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results = []

        for query in queries:
            _results = self.chat.browser_results(query, self.k)
            results = None
            is_dict_results = isinstance(_results, dict) and "results" in _results
            if is_dict_results:
                results = _results["results"]
            else:
                if not isinstance(_results, dict):
                    # print type of _results
                    print(f"Invalid type of results: {type(_results)}")
                else:
                    print("missing results key in _results {results.keys()}")
                continue

            for d in results:
                # assert d is dict
                if not isinstance(d, dict):
                    print(f"Invalid result: {d}")
                    continue

                try:
                    # ensure keys are present
                    url = d.get("url", None)
                    title = d.get("title", None)
                    description = d.get("description", None)
                    snippets = d.get("snippets", None)

                    # raise exception of missing key(s)
                    if not all([url, title, description, snippets]):
                        raise ValueError(f"Missing key(s) in result: {d}")
                    if self.is_valid_source(url) and url not in exclude_urls:
                        result = {
                            "url": url,
                            "title": title,
                            "description": description,
                            "snippets": snippets,
                        }
                        collected_results.append(result)
                    else:
                        print(f"invalid source {url} or url in exclude_urls")
                except Exception as e:
                    print(f"Error occurs when processing {result=}: {e}")
                    logging.error(f"Error occurs when searching query {query}: {e}")

        return collected_results
