# LangChain Experiment with Question and Answer Generation

## Overview

This experiment demonstrates the use of the LangChain framework to generate a structured JSON format of questions and answers from a given text using an LLM (Large Language Model). The task was to convert text data into a JSON object where each entry contains a question and its corresponding answer. This was achieved using LangChain's LLMChain with a specific prompt template designed for this purpose.

## Project Setup

### Requirements

- Python 3.7 or higher
- LangChain and its dependencies
- A suitable LLM model (`llama3.1` in this case)
- PyPDFLoader for loading and splitting PDF documents

### Installation

Ensure that you have installed all the required libraries:

```bash
pip install langchain langchain-community
```

### File Structure

- `main.py`: Contains the script for loading the PDF, processing the text using LangChain, and generating the JSON output.
- `Question Bank 1 (Tan et al 2nd Edition).pdf`: The input PDF file containing text to be converted to a JSON format.

## Experiment Steps

1. **Document Loading**: 
   - The PDF file `Question Bank 1 (Tan et al 2nd Edition).pdf` was loaded using `PyPDFLoader`.
   - The document was split into individual pages for processing.

2. **Embeddings Generation**:
   - `OllamaEmbeddings` was used to generate embeddings for the text content from each page of the document.

3. **Prompt Template Creation**:
   - A `PromptTemplate` was defined to instruct the LLM to generate a list of questions and answers from the provided text. The prompt explicitly states that the output should be in JSON format.

   ```python
   template: str = """You are a teacher who is creating a list of questions and answers. 
   You are generating a usable json from the given text. The json has an array of objects with 
   two keys: question, answer. Given text is: {text} json: """
   ```

4. **LLM Model Setup**:
   - The `ChatOllama` model (`llama3.1`) was chosen for this task. The temperature parameter was set to 0.7 to balance creativity and coherence in the output.

5. **Chain Initialization and Prediction**:
   - An `LLMChain` was initialized using the defined prompt template and the LLM model.
   - The `predict` function was called with the text content of the first page to generate the output.

## Output

Upon running the experiment, the following JSON output was generated:

```json
[
  {
    "question": "For each data set given below, give speciﬁc examples of classiﬁcation, clustering, association rule mining, and anomaly detection tasks that can be performed on the data. For each task, state how the data matrix should be constructed (i.e., specify the rows and columns of the matrix).",
    "answer": ""
  },
  {
    "question": "(a) Ambulatory Medical Care data1, which contains the demographic and medical visit information for each patient (e.g., gender, age, duration of visit, physician’s diagnosis, symptoms, medication, etc)",
    "answer": [
      {
        "task": "Classiﬁcation",
        "description": "Task: Diagnose whether a patient has a disease.",
        "rows": "Patient",
        "columns": "Patient’s demographic and hospital visit information (e.g., symptoms), along with a class attribute that indicates whether the patient has the disease."
      },
      {
        "task": "Clustering",
        "description": "Task: Find groups of patients with similar medical conditions",
        "rows": "A patient visit",
        "columns": "List of medical conditions of each patient"
      },
      {
        "task": "Association rule mining",
        "description": "Task: Identify the symptoms and medical conditions that co-occur together frequently",
        "rows": "A patient visit",
        "columns": "List of symptoms and diagnosed medical conditions of the patient"
      },
      {
        "task": "Anomaly detection",
        "description": "Task: Identify healthy looking patients with rare medical disorders",
        "rows": "A patient visit",
        "columns": "List of demographic attributes, symptoms, and medical test results of the patient"
      }
    ]
  }
]
```

This output shows the structured JSON format containing questions derived from the text, along with detailed answers structured into tasks, descriptions, rows, and columns.
