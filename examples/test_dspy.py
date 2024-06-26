import dspy
from rich import print
from run_storm_wiki_gpt import GPT_3_5_TURBO, GPT_4O, get_openai_kwargs

from lm import OpenAIModel

openai_kwargs = get_openai_kwargs()

engine = OpenAIModel(model=GPT_4O, max_tokens=500, **openai_kwargs)


# from tetsts, the urls are not valid
class FindRelatedTopic(dspy.Signature):
    """I'm writing a Economist article for a topic mentioned below. Please identify and recommend some Economist articles on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Economist articles for similar topics.
    Please list the urls in separate lines."""

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    related_topics = dspy.OutputField(format=str)


topic = "Fusion vs French cuisine"
find_related_topic = dspy.ChainOfThought(FindRelatedTopic)

with dspy.settings.context(lm=engine):
    related_topics = find_related_topic(topic=topic)

    print(related_topics)
