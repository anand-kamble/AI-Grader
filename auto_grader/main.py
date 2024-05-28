import os
from typing import List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langsmith import traceable

load_dotenv()


# @traceable
def format_prompt(subject: str) -> List[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a Data mining Instructor",
        },
        {
            "role": "user",
            "content": f"What's a good name for a store that sells {subject}?",
        },
    ]


llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
    },
)


# @traceable(run_type="llm")
def invoke_llm(messages: List[dict[str, str]]):

    return llm.invoke(messages)


# @traceable
def parse_output(response):
    return response


# @traceable
def run_pipeline():
    messages = format_prompt("colorful socks")
    # response = invoke_llm(messages)
    # return parse_output(response)


def main():
    print("Running pipeline...")
    res = run_pipeline()
    print(res)


if __name__ == "__main__":
    main()
