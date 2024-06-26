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


_args = {
    "output_dir": "~/Downloads/storm",
    "max_thread_num": 3,
    "retriever": "openai",
    "do_research": True,
    "do_generate_outline": True,
    "do_generate_article": False,
    "do_polish_article": False,
    "max_conv_turn": 3,
    "max_perspective": 3,
    "search_top_k": 3,
    "max_thread_num": 3,
}

# convert _args to namespace
args = type("args", (object,), _args)()

topic = "WordPress vs Wix"


def main(args):
    load_api_key(toml_file_path="secrets.toml")
    lm_configs = STORMWikiLMConfigs()
    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_provider": os.getenv("OPENAI_API_TYPE"),
        "temperature": 1.0,
        "top_p": 0.9,
    }

    if os.getenv("OPENAI_API_TYPE") == "azure":
        openai_kwargs["api_base"] = os.getenv("AZURE_API_BASE")
        openai_kwargs["api_version"] = os.getenv("AZURE_API_VERSION")

    # STORM is a LM system so different components can be powered by different models.
    # For a good balance between cost and quality, you can choose a cheaper/faster model for conv_simulator_lm
    # which is used to split queries, synthesize answers in the conversation. We recommend using stronger models
    # for outline_gen_lm which is responsible for organizing the collected information, and article_gen_lm
    # which is responsible for generating sections with citations.

    # Using constants to instantiate OpenAIModel objects
    conv_simulator_lm = OpenAIModel(
        model=GPT_3_5_TURBO, max_tokens=500, **openai_kwargs
    )
    question_asker_lm = OpenAIModel(
        model=GPT_3_5_TURBO, max_tokens=500, **openai_kwargs
    )
    outline_gen_lm = OpenAIModel(model=GPT_4O, max_tokens=400, **openai_kwargs)
    article_gen_lm = OpenAIModel(model=GPT_4O, max_tokens=700, **openai_kwargs)
    article_polish_lm = OpenAIModel(model=GPT_4O, max_tokens=4000, **openai_kwargs)

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
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

    runner.run(
        topic=topic,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
    )
    runner.post_run()
    runner.summary()


if __name__ == "__main__":
    main(args)
