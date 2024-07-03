import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import dspy
from run_storm_wiki_gpt import GPT_3_5_TURBO, GPT_4O, get_openai_kwargs

from interface import Retriever
from rm import OpenAIBrowserSearch
from storm_wiki.modules.retriever import StormRetriever
from storm_wiki.modules.storm_dataclass import DialogueTurn, StormInformation

sys.path.append("./src")
from lm import OpenAIModel
from rm import OpenAIBrowserSearch
from utils import ArticleTextProcessing

_args = {
    "output_dir": "~/Downloads/storm",
    "max_thread_num": 3,
    "retriever": "openai",
    "do_research": True,
    "do_generate_outline": True,
    "do_generate_article": False,
    "do_polish_article": False,
    "max_conv_turn": 3,
    "max_perspective": 5,
    "search_top_k": 3,
    "max_thread_num": 3,
}

# convert _args to namespace
args = type("args", (object,), _args)()

topic = "Capitalism is better than socialism"
topic = "Wix is better than WordPress "
DEBATE_ROLES = ["proposer", "opposer"]
DEBATE_ROLES = ["opposer", "proposer"]


def get_topic_output_dir(topic: str):
    topic_dir_name = topic.replace(" ", "_").replace("/", "_")
    topic_output_dir = os.path.join(args.output_dir, topic_dir_name)
    # expand user path
    topic_output_dir = os.path.expanduser(topic_output_dir)
    os.makedirs(topic_output_dir, exist_ok=True)
    return topic_output_dir


def get_debater_engine():
    return OpenAIModel(model=GPT_3_5_TURBO, max_tokens=500, **get_openai_kwargs())


def get_topic_expert_engine():
    return OpenAIModel(model=GPT_4O, max_tokens=500, **get_openai_kwargs())


def get_rm():
    rm = OpenAIBrowserSearch(openai_api_key=os.getenv("OPENAI_API"))
    return StormRetriever(rm, k=args.search_top_k)


class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced debater. You are articulate and persuasive, always aiming to produce strong arguments with well-researched information.
    Now, you are chatting with an expert to get more useful information to support your stance and to rebut arguments. Ask insightful questions to gather valuable evidence and data.
    When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask one question at a time and don't repeat questions. Your questions should relate to rebutting the argument from the other party and to strengthen your  position on the topic you are debating.
    If you are the opposer, ask questions to argue AGAINST the topic.
    If you are the proposer, ask questions to argue FOR the topic.
    """

    topic = dspy.InputField(prefix="Debate topic: ", format=str)
    argument = dspy.InputField(prefix="Argument to rebut: ", format=str)
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


class DebateWriter(dspy.Module):
    """Perspective-guided question asking in conversational setup.

    The asked question will be used to start a next round of information seeking."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        self.engine = engine

    def forward(
        self,
        topic: str,
        persona: str,
        argument: str,
        dialogue_turns: List[DialogueTurn],
    ):
        conv = []
        for turn in dialogue_turns[:-4]:
            conv.append(
                f"{persona}: {turn.user_utterance}\nExpert: Omit the answer here due to space limit."
            )
        for turn in dialogue_turns[-4:]:
            conv.append(
                f"{persona}: {turn.user_utterance}\nExpert: {ArticleTextProcessing.remove_citations(turn.agent_utterance)}"
            )
        conv = "\n".join(conv)
        conv = conv.strip() or "N/A"
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)

        with dspy.settings.context(lm=self.engine):
            question = self.ask_question_with_persona(
                topic=topic, persona=persona, conv=conv, argument=argument
            ).question
        print(f"==> {persona}_writer: {question=}")
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
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, topic: str, question: str, ground_truth_url: str = ""):
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

        # print(f"==> TopicExpert: {queries=} {answer=}")
        return dspy.Prediction(
            queries=queries, searched_results=searched_results, answer=answer
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
        self.debate_writer = DebateWriter(engine=get_debater_engine())
        self.topic_expert = TopicExpert(
            engine=get_topic_expert_engine(),
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=get_rm(),
        )

        self.max_turn = max_turn

    def forward(self, topic: str):
        """
        topic: The topic to debate
        """
        dlg_histories: Dict[str, List[DialogueTurn]] = {"proposer": [], "opposer": []}
        # ==> here is the conversation user_utterance=question
        # expert_output=answer
        argument = None
        combined_dlg_history = []
        for _ in range(self.max_turn):
            for debater in DEBATE_ROLES:
                dlg_history = dlg_histories[debater]
                user_utterance = self.debate_writer(
                    topic=topic,
                    persona=debater,
                    argument=argument,
                    dialogue_turns=dlg_history,
                ).question
                if user_utterance == "":
                    logging.error("Simulated Debate writer utterance is empty.")
                    break
                if user_utterance.startswith("Thank you so much for your help!"):
                    break
                expert_output = self.topic_expert(topic=topic, question=user_utterance)

                argument = expert_output.answer

                dlg_turn = DialogueTurn(
                    agent_utterance=expert_output.answer,
                    user_utterance=user_utterance,
                    search_queries=expert_output.queries,
                    search_results=expert_output.searched_results,
                )
                dlg_history.append(dlg_turn)
                combined_dlg_history.append(dlg_turn)

        return dspy.Prediction(dlg_history=combined_dlg_history)


def _run_conversation(
    conv_simulator,
    topic,
) -> List[Tuple[str, List[DialogueTurn]]]:
    """
    Executes multiple conversation simulations concurrently, each with a different persona,
    and collects their dialog histories. The dialog history of each conversation is cleaned
    up before being stored.

    Parameters:
        conv_simulator (callable): The function to simulate conversations. It must accept four
            parameters: `topic`, `ground_truth_url`, `persona`, and `callback_handler`, and return
            an object that has a `dlg_history` attribute.
        topic (str): The topic of conversation for the simulations.
        ground_truth_url (str): The URL to the ground truth data related to the conversation topic.
        considered_personas (list): A list of personas under which the conversation simulations
            will be conducted. Each persona is passed to `conv_simulator` individually.
        callback_handler (callable): A callback function that is passed to `conv_simulator`. It
            should handle any callbacks or events during the simulation.

    Returns:
        list of tuples: A list where each tuple contains a persona and its corresponding cleaned
        dialog history (`dlg_history`) from the conversation simulation.
    """

    conversations = []

    def run_conv():
        return conv_simulator(
            topic=topic,
        )

    conversations.append(run_conv())

    return conversations


def main():
    conv_simulator = ConvSimulator(
        max_search_queries_per_turn=3, search_top_k=3, max_turn=3
    )
    conversation = _run_conversation(conv_simulator, topic)
    topic_output_dir = get_topic_output_dir(topic)
    with open(f"{topic_output_dir}/debate.txt", "w") as f:
        f.write(f"{conversation=}")


if __name__ == "__main__":
    main()
