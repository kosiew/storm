import logging
import sys
from typing import List, Tuple, Union

import dspy
from run_debate import TopicExpert, get_rm, get_topic_expert_engine
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
    "output_dir": "~/Downloads/brainstorm",
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

topic = "Interesting topics for a Swiss watch blog"


def get_thinker_engine():
    return OpenAIModel(model=GPT_3_5_TURBO, max_tokens=500, **get_openai_kwargs())


class GenPersona(dspy.Signature):
    """You need to select a group of participants who will work together for a brainstorming session.
    Each of them represents a different perspective, role, or affiliation related to this topic.
    For each participant, add a description of what they will focus on.
    Give your answer in the following format: 1. short summary of participant1: description\n2. short summary of participant2: description\n...
    """

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    personas = dspy.OutputField(format=str)


class CreateBrainstormPersona(dspy.Module):
    """Discover different perspectives of researching the topic."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.engine = engine

    def forward(self, topic: str, draft=None):
        with dspy.settings.context(lm=self.engine):

            gen_persona_output = self.gen_persona(topic=topic).personas

        personas = []
        for s in gen_persona_output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", s)
            if match:
                personas.append(match.group(1))

        print(f"==> Generated personas for the topic '{topic}': {personas=}")
        sorted_personas = personas

        return dspy.Prediction(personas=personas, raw_personas_output=sorted_personas)


class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced creative thinker in your field. You are articulate and persuasive, always aiming to produce compelling, practical and unconventional ideas.
    Now, you are chatting with an expert to get more useful information to present your ideas. Ask insightful questions to gather valuable evidence and data.
    When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask one question at a time and don't repeat questions.
    """

    topic = dspy.InputField(prefix="Debate topic: ", format=str)
    persona = dspy.InputField(prefix="Your role in this debate", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class CreativeThinker(dspy.Module):
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
                topic=topic, persona=persona, conv=conv
            ).question
        print(f"==> {persona}_writer: {question=}")
        return dspy.Prediction(question=question)


class ConvSimulator(dspy.Module):
    """Simulate a conversation between a proposer and an opposer"""

    def __init__(
        self,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_turn: int,
    ):
        super().__init__()
        self.thinker = CreativeThinker(engine=get_thinker_engine())
        self.create_personas = CreateBrainstormPersona(engine=get_topic_expert_engine())

        self.topic_expert = TopicExpert(
            engine=get_topic_expert_engine(),
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=get_rm(),
        )

        self.max_turn = max_turn

    def forward(self, topic: str):
        """
        topic: The topic to brainstorm
        """
        dlg_history: List[DialogueTurn] = []
        personas = self.create_personas(topic=topic).personas

        for _ in range(self.max_turn):
            for persona in personas:
                user_utterance = self.thinker(
                    topic=topic,
                    persona=persona,
                    dialogue_turns=dlg_history,
                ).question
                if user_utterance == "":
                    logging.error("Simulated thinker utterance is empty.")
                    break
                if user_utterance.startswith("Thank you so much for your help!"):
                    break
                expert_output = self.topic_expert(topic=topic, question=user_utterance)

                dlg_turn = DialogueTurn(
                    agent_utterance=expert_output.answer,
                    user_utterance=user_utterance,
                    search_queries=expert_output.queries,
                    search_results=expert_output.searched_results,
                )
                dlg_history.append(dlg_turn)

        return dspy.Prediction(dlg_history=dlg_history)


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
    print(f"{conversation=}")


if __name__ == "__main__":
    main()
