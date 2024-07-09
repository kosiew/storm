import os
import sys
from argparse import ArgumentParser

sys.path.append("./src")
from lm import OpenAIModel
from rm import BingSearch, OpenAIBrowserSearch, YouRM
from storm_wiki.engine import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)
from utils import load_api_key

GPT_3_5_TURBO = "gpt-3.5-turbo"
GPT_4O = "gpt-4o"
"""
STORM Wiki pipeline powered by GPT-3.5/4 and You.com search engine.
You need to set up the following environment variables to run this script:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_API_TYPE: OpenAI API type (e.g., 'openai' or 'azure')
    - AZURE_API_BASE: Azure API base URL if using Azure API
    - AZURE_API_VERSION: Azure API version if using Azure API
    - YDC_API_KEY: You.com API key; or, BING_SEARCH_API_KEY: Bing Search API key

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""


def main(args):
    openai_kwargs = get_openai_kwargs()

    lm_configs = STORMWikiLMConfigs()

    # STORM is a LM system so different components can be powered by different models.
    # For a good balance between cost and quality, you can choose a cheaper/faster model for conv_simulator_lm
    # which is used to split queries, synthesize answers in the conversation. We recommend using stronger models
    # for outline_gen_lm which is responsible for organizing the collected information, and article_gen_lm
    # which is responsible for generating sections with citations.

    # Using constants to instantiate OpenAIModel objects
    conv_simulator_lm = OpenAIModel(
        model=GPT_3_5_TURBO, max_tokens=500, **openai_kwargs
    )
    question_asker_lm = OpenAIModel(model=GPT_4O, max_tokens=500, **openai_kwargs)
    outline_gen_lm = OpenAIModel(model=GPT_4O, max_tokens=400, **openai_kwargs)
    article_gen_lm = OpenAIModel(model=GPT_4O, max_tokens=700, **openai_kwargs)
    article_polish_lm = OpenAIModel(model=GPT_4O, max_tokens=4000, **openai_kwargs)

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    # expand user path for output_dir
    output_dir = os.path.expanduser(args.output_dir)
    engine_args = STORMWikiRunnerArguments(
        output_dir=output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # STORM is a knowledge curation system which consumes information from the retrieval module.
    # Currently, the information source is the Internet and we use search engine API as the retrieval module.
    if args.retriever == "bing":
        rm = BingSearch(
            bing_search_api=os.getenv("BING_SEARCH_API_KEY"), k=engine_args.search_top_k
        )
    elif args.retriever == "you":
        rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
    elif args.retriever == "openai":
        rm = OpenAIBrowserSearch(
            openai_api_key=os.getenv("OPENAI_API_KEY"), k=engine_args.search_top_k
        )

    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    topic = input("Topic: ")
    if topic:
        run_topic(args, runner, topic)
    else:
        try_topics_file(args, runner)


def try_topics_file(args, runner):
    # expand user path for topics_file
    topics_file = os.path.expanduser(args.topics_file)
    with open(topics_file, "r") as f:
        topics = f.readlines()
    for topic in topics:
        topic = topic.strip()
        if topic:
            run_topic(args, runner, topic)

    # remove contents of topics file
    with open(topics_file, "w") as f:
        f.write("")


def run_topic(args, runner, topic):
    runner.run(
        topic=topic,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
    )
    runner.post_run()
    runner.summary()


def get_openai_kwargs():
    load_api_key(toml_file_path="secrets.toml")
    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_provider": os.getenv("OPENAI_API_TYPE"),
        "temperature": 1.0,
        "top_p": 0.9,
    }
    if os.getenv("OPENAI_API_TYPE") == "azure":
        openai_kwargs["api_base"] = os.getenv("AZURE_API_BASE")
        openai_kwargs["api_version"] = os.getenv("AZURE_API_VERSION")

    return openai_kwargs


if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/Downloads/storm",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="Maximum number of threads to use. The information seeking part and the article generation"
        "part can speed up by using multiple threads. Consider reducing it if keep getting "
        '"Exceed rate limit" error when calling LM API.',
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "openai"],
        help="The search engine API to use for retrieving information.",
    )
    # stage of the pipeline
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="If True, simulate conversation to research the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic; otherwise, load the results.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If True, polish the article by adding a summarization section and (optionally) removing "
        "duplicate content.",
    )
    # hyperparameters for the pre-writing stage
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="Maximum number of questions in conversational question asking.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Maximum number of perspectives to consider in perspective-guided question asking.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="Top k search results to consider for each search query.",
    )
    # hyperparameters for the writing stage
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=3,
        help="Top k collected references for each section title.",
    )
    parser.add_argument(
        "--remove-duplicate",
        action="store_true",
        help="If True, remove duplicate content from the article.",
    )

    # add argument for a topics file
    parser.add_argument(
        "--topics-file",
        type=str,
        default="~/Downloads/storm/topics.txt",
        help="File containing topics to run the pipeline on.",
    )

    main(parser.parse_args())
