import os
import pickle

from paperqa import Docs, QueryRequest, Settings, ask
from paperqa.agents.models import AnswerResponse
from paperqa.settings import AgentSettings
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "i-am-not-used-but-must-be-here"

local_llm_config = {
    "model_list": [
        {
            "model_name": "ollama/llama3.1",
            "litellm_params": {
                "model": "ollama/llama3.1",
                "api_base": "http://localhost:11434",
            },
        }
    ]
}


# Specify the pickle file and the directory containing your PDF files
pickle_file = "pqa_docs_cache.pkl"
docs_dir = "solutions_questions"  # Replace with your directory path

# Check if the pickle file exists
if os.path.exists(pickle_file):
    # Load the Docs object from the pickle file
    with open(pickle_file, "rb") as f:
        docs = pickle.load(f)
    print("Loaded Docs object from pickle.")
else:
    # Create a new Docs object
    docs = Docs()
    print("Creating new Docs object.")
    pdf_files: list[str] = [
        f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")
    ]

    # Iterate over all files in the directory
    for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(docs_dir, filename)
            try:
                # Add each PDF file to the Docs object
                docs.add(
                    filepath,
                    settings=Settings(
                        llm="ollama/llama3.1",
                        llm_config=local_llm_config,
                        summary_llm="ollama/llama3.1",
                        summary_llm_config=local_llm_config,
                        temperature=0.2,
                        embedding="ollama/llama3.1",
                    ),
                )
                print(f"Added {filename} to Docs.")
            except Exception as e:
                print(f"Failed to add {filename}: {e}")

    # Save the Docs object to a pickle file for future use
    with open(pickle_file, "wb") as f:
        pickle.dump(docs, f)
    print("Saved Docs object to pickle.")


# answer: AnswerResponse = ask(
#     query="What is K means?",
#     settings=Settings(
#         llm="ollama/llama3.1",
#         llm_config=local_llm_config,
#         summary_llm="ollama/llama3.1",
#         summary_llm_config=local_llm_config,
#         paper_directory="solutions_questions",
#         temperature=0.2,
#     ),
# )

answer = docs.query(
    query="What is K means?",
    settings=Settings(
        llm="ollama/llama3.1",
        llm_config=local_llm_config,
        summary_llm="ollama/llama3.1",
        summary_llm_config=local_llm_config,
        temperature=0.2,
        embedding="ollama/llama3.1",
    ),
)

print(answer)
