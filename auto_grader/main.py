# %%
import os
from typing import Any, List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langsmith import traceable
from utils import constants, pickel_loader

load_dotenv()


def load_correct_answers():
    """
    This function loads and returns the correct answers from a predefined path.

    Returns:
    Any: The correct answers loaded from the pickle file.
    """
    return pickel_loader(constants.CORRECT_ANSWER_PATH)


def load_student_answers(path: str) -> Any:
    """
    This function loads and returns the student answers from a given path.

    Parameters:
    path (str): The path of the pickle file to be loaded.

    Returns:
    Any: The student answers loaded from the pickle file.
    """
    return pickel_loader(path)


def format_prompt(
    correct_answer: str, student_answer: str, context: str = "No context provided"
) -> List[dict[str, str]]:
    """
    This function formats the grading prompt for an assignment grader.

    Parameters:
    correct_answer (str): The correct answer for the assignment.
    student_answer (str): The student's answer for the assignment.
    context (str, optional): The context for the assignment. Defaults to "No context provided".

    Returns:
    List[dict[str, str]]: A list of dictionaries containing the formatted prompts for the system and the user.
    """
    return [
        {
            "role": "system",
            "content": f"You are a assignment grader, you grade explainations from 0 to 10.\nContext:{context} \nThe expected explaination is:{correct_answer}",
        },
        {
            "role": "user",
            "content": f"Please grade the following answer: {student_answer}.",
        },
    ]


# The @traceable decorator is used to track the execution of the function.
# It can be used for logging, debugging, or performance monitoring.
@traceable
def invoke_llm(messages: List[dict[str, str]]) -> Any:
    """
    This function invokes the HuggingFacePipeline with a specific model and returns the result of the invocation.

    The model used is "microsoft/Phi-3-mini-128k-instruct" and the task is "text-generation". The maximum length of the generated text is 1024.

    Parameters:
    messages (List[dict[str, str]]): A list of dictionaries containing the prompts for the system and the user.

    Returns:
    Any: The result of the invocation of the HuggingFacePipeline.
    """
    # Initialize the HuggingFacePipeline with the specified model, task, and pipeline arguments
    llm = HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3-mini-128k-instruct",  # The model to be used
        task="text-generation",  # The task to be performed
        pipeline_kwargs={
            "max_length": 1024,  # The maximum length of the generated text
        },
        device_map="auto",  # Automatically select the device to run the model on
    )

    # Invoke the HuggingFacePipeline with the provided messages and return the result
    return llm.invoke(messages)


def main():
    """
    This is the main function that runs the grading process.

    It first loads the correct answers and the student's answers. Then, it formats the grading prompt and invokes the language model to grade the student's answer. Finally, it prints the grading result.
    """
    # Print a message to indicate the start of the grading process
    print("Running grader ...")

    # Load the correct answers
    correct_answers = load_correct_answers()

    # Load the student's answers
    # Path is relative to the ./run.sh script.
    student_answers = load_student_answers("./student_code/answers.pkl")

    # Define the context for the grading prompt
    context = "K-Means and Agglomerative Clustering are two popular unsupervised machine learning algorithms used for data clustering. While both methods have their strengths and weaknesses, K-Means is generally more efficient for large datasets due to its linear time complexity compared to Agglomerative Clustering's quadratic time complexity. This efficiency allows K-Means to handle large datasets more quickly and effectively, making it a preferred choice for production-scale systems. Additionally, K-Means is more scalable and can handle datasets with over 10,000 data points, whereas Agglomerative Clustering can become computationally expensive and impractical for such large datasets."

    # Format the grading prompt
    messages = format_prompt(
        correct_answers["question1"]["(c) explain"],
        student_answers["question1"]["(c) explain"],
        context=context,
    )

    # Invoke the language model to grade the student's answer
    response = invoke_llm(messages)
    print("Response is: ", response)


if __name__ == "__main__":
    main()
