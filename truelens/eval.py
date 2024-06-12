# %%
# pip install trulens_eval llama_index openai
# pip install "litellm>=1.25.2"

# %%

from imghdr import tests

import numpy as np
import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.app import App
from trulens_eval.feedback.provider import LiteLLM

# %%
embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embeddings

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)

generator_llm = Ollama(model="phi3:latest")
query_engine = index.as_query_engine(llm=generator_llm)


# %%
provider = LiteLLM(
    model_engine="ollama/phi3:latest",
    endpoint="http://localhost:11434",
    kwargs={
        "set_verbose": True
    },  # I have added this so that its easie to debug the LLM.
)
context = App.select_context(query_engine)


# %%
# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons)
    .on(context.collect())  # collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = Feedback(provider.relevance).on_input_output()
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

# %%

from trulens_eval import TruLlama

tru_query_engine_recorder = TruLlama(
    query_engine,
    app_id="LlamaIndex_App1",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)


# %%
testset = pd.read_csv("../testset.csv")

testset_dict = dict()
testset_dict["question"] = list(testset["question"])
testset_dict["ground_truth"] = list(testset["ground_truth"])

# %%
with tru_query_engine_recorder as recording:
    # query_engine.query("What did the author do growing up?")
    query_engine.query(testset_dict)
    # %%
    # The record of the app invocation can be retrieved from the `recording`:

    rec = recording.get()  # use .get if only one record
    # recs = recording.records # use .records if multiple

    # %%
    tru = Tru()
    # tru.run_dashboard()

    # %%

    # The results of the feedback functions can be rertireved from
    # `Record.feedback_results` or using the `wait_for_feedback_result` method. The
    # results if retrieved directly are `Future` instances (see
    # `concurrent.futures`). You can use `as_completed` to wait until they have
    # finished evaluating or use the utility method:

    for feedback, feedback_result in rec.wait_for_feedback_results().items():
        print(feedback.name, feedback_result.result)

    # See more about wait_for_feedback_results:
    # help(rec.wait_for_feedback_results)

    records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])

    records.head()

    # %%
    tru.run_dashboard()
