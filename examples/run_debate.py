import os
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, Union

import dspy
from run_storm_wiki_gpt import GPT_3_5_TURBO, GPT_4O, get_openai_kwargs

from rm import OpenAIBrowserSearch

sys.path.append("./src")
from lm import OpenAIModel
from rm import BingSearch, OpenAIBrowserSearch, YouRM
from storm_wiki.engine import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)
from utils import load_api_key

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

topic = "WordPress is better than Wix"

rm_proposer = OpenAIBrowserSearch(openai_api_key=os.getenv("OPENAI_API_KEY"))
rm_opposer = OpenAIBrowserSearch(openai_api_key=os.getenv("OPENAI_API_KEY"))


class AskQuestion(dspy.Signature):
    """You are an experienced debater. You are chatting with an expert to get information for the topic you are debating. Ask good questions to get more useful information relevant to the topic.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write.
    """

    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class AskQuestionWithPersona(dspy.Signature):
    """You are a experienced debater. You are articulate and persuasive, always aiming to strengthen your arguments with well-researched information. Now, you are chatting with an expert to get more useful information to support your stance. Ask insightful questions to gather valuable evidence and data.
    When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask one question at a time and don't repeat questions. Your questions should be related to the topic you are debating.
    """

    topic = dspy.InputField(prefix="Debate topic: ", format=str)
    persona = dspy.InputField(prefix="Your role in this debate", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class QuestionToQuery(dspy.Signature):
    """You want to answer the question using Google search. What do you type in the search box?
    Write the queries you will use in the following format:
    - query 1
    - query 2
    ...
    - query n"""

    topic = dspy.InputField(prefix="Topic you are discussing about: ", format=str)
    question = dspy.InputField(prefix="Question you want to answer: ", format=str)
    queries = dspy.OutputField(format=str)


class AnswerQuestion(dspy.Signature):
    """You are an expert who can use information effectively. You are chatting with a debater writer who wants to prepare his argument on topic you know. You have gathered the related information and will now use the information to form a response.
    Make your response as informative as possible and make sure every sentence is supported by the gathered information. If [Gathered information] is not related to he [Topic] and [Question], output "Sorry, I don't have enough information to answer the question.".
    """

    topic = dspy.InputField(prefix="Topic you are discussing about:", format=str)
    conv = dspy.InputField(prefix="Question:\n", format=str)
    info = dspy.InputField(prefix="Gathered information:\n", format=str)
    answer = dspy.OutputField(
        prefix="Now give your response. (Try to use as many different sources as possible and add do not hallucinate.)\n",
        format=str,
    )


class Debater(dspy.Module):
    """Perspective-guided question asking in conversational setup.

    The asked question will be used to start a next round of information seeking."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        self.ask_question = dspy.ChainOfThought(AskQuestion)
        self.engine = engine

    def forward(
        self,
        topic: str,
        persona: str,
        dialogue_turns: List[DialogueTurn],
        draft_page=None,
    ):
        conv = []
        for turn in dialogue_turns[:-4]:
            conv.append(
                f"You: {turn.user_utterance}\nExpert: Omit the answer here due to space limit."
            )
        for turn in dialogue_turns[-4:]:
            conv.append(
                f"You: {turn.user_utterance}\nExpert: {ArticleTextProcessing.remove_citations(turn.agent_utterance)}"
            )
        conv = "\n".join(conv)
        conv = conv.strip() or "N/A"
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)

        with dspy.settings.context(lm=self.engine):
            if persona is not None and len(persona.strip()) > 0:
                question = self.ask_question_with_persona(
                    topic=topic, persona=persona, conv=conv
                ).question
            else:
                question = self.ask_question(
                    topic=topic, persona=persona, conv=conv
                ).question
        print(f"==> wiki_writer: {question=}")
        return dspy.Prediction(question=question)


class TopicExpert(dspy.Module):
    """Answer questions using search-based retrieval and answer generation. This module conducts the following steps:
    1. Generate queries from the question.
    2. Search for information using the queries.
    3. Filter out unreliable sources.
    4. Generate an answer using the retrieved information.
    """

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries: int,
        search_top_k: int,
        retriever: Retriever,
    ):
        super().__init__()
        self.generate_queries = dspy.Predict(QuestionToQuery)
        self.retriever = retriever
        self.retriever.update_search_top_k(search_top_k)
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, topic: str, question: str, ground_truth_url: str):
        with dspy.settings.context(lm=self.engine):
            # Identify: Break down question into queries.
            queries = self.generate_queries(topic=topic, question=question).queries
            queries = [
                q.replace("-", "").strip().strip('"').strip('"').strip()
                for q in queries.split("\n")
            ]
            queries = queries[: self.max_search_queries]
            # Search
            searched_results: List[StormInformation] = self.retriever.retrieve(
                list(set(queries)), exclude_urls=[ground_truth_url]
            )
            if len(searched_results) > 0:
                # Evaluate: Simplify this part by directly using the top 1 snippet.
                info = ""
                for n, r in enumerate(searched_results):
                    info += "\n".join(f"[{n + 1}]: {s}" for s in r.snippets[:1])
                    info += "\n\n"

                info = ArticleTextProcessing.limit_word_count_preserve_newline(
                    info, 1000
                )

                try:
                    answer = self.answer_question(
                        topic=topic, conv=question, info=info
                    ).answer
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )
                except Exception as e:
                    logging.error(f"Error occurs when generating answer: {e}")
                    answer = "Sorry, I cannot answer this question. Please ask another question."
            else:
                # When no information is found, the expert shouldn't hallucinate.
                answer = "Sorry, I cannot find information for this question. Please ask another question."

        print(f"==> TopicExpert: {queries=} {answer=}")
        return dspy.Prediction(
            queries=queries, searched_results=searched_results, answer=answer
        )


class DialogueTurn:
    def __init__(
        self,
        proposer_argument: str = None,
        opposer_argument: str = None,
    ):
        self.proposer_argument = proposer_argument
        self.opposer_argument = opposer_argument

    def log(self):
        """
        Returns a json object that contains all information inside `self`
        """

        return OrderedDict(
            {
                "proposer_argument": self.proposer_argument,
                "opposer_argument": self.opposer_argument,
            }
        )


class ConvSimulator(dspy.Module):
    """Simulate a conversation between a proposer and an opposer"""

    def __init__(
        self,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_turn: int,
    ):
        super().__init__()
        self.proposer = TopicExpert(
            engine=OpenAIModel(model=GPT_4O, max_tokens=500, **openai_kwargs),
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=rm_proposer,
        )
        self.opposer = TopicExpert(
            engine=OpenAIModel(model=GPT_4O, max_tokens=500, **openai_kwargs),
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=rm_opposer,
        )
        self.max_turn = max_turn

    def forward(self, topic: str):
        """
        topic: The topic to debate
        """
        dlg_history: List[DialogueTurn] = []
        # ==> here is the conversation user_utterance=question
        # expert_output=answer
        opposer_argument = None
        for _ in range(self.max_turn):
            proposer_argument = self.proposer(topic=topic, dialogue_turns=dlg_history)
            expert_output = self.topic_expert(
                topic=topic, question=user_utterance, ground_truth_url=ground_truth_url
            )
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=user_utterance,
                search_queries=expert_output.queries,
                search_results=expert_output.searched_results,
            )
            dlg_history.append(dlg_turn)
            callback_handler.on_dialogue_turn_end(dlg_turn=dlg_turn)

        return dspy.Prediction(dlg_history=dlg_history)


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
    proposer_lm = OpenAIModel(model=GPT_3_5_TURBO, max_tokens=500, **openai_kwargs)
    opposer_lm = OpenAIModel(model=GPT_3_5_TURBO, max_tokens=500, **openai_kwargs)

    adjudicator_lm = OpenAIModel(model=GPT_3_5_TURBO, max_tokens=500, **openai_kwargs)

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
