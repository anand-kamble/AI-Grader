
# AI Grader

This project aims to develop a completely automated grading system using a Large Language Model (LLM).

## Execution
To set up and run the project, follow these steps:

1. **Activate the Poetry shell:**
   ```bash
   poetry shell
   ```
2. **Install the dependencies:**
   ```bash
   poetry install
   ```
3. **Run the application:**
   ```bash
   ./run.sh
   ```

   To generate the test set, you can pass the `--testset` argument to the script:
   ```bash
   ./run.sh --testset
   ```

   To use TruLens, you can pass the `--trulens` argument to the script:
   ```bash
   ./run.sh --trulens
   ```

## Details

### Basic Flow of the Program

1. **Loading Answers:**
   - The answers are loaded from pickle files generated by Python scripts used for grading.
   
2. **Creating a Prompt:**
   - A prompt is created using the loaded answers. This prompt includes the correct answer, context, and the student's answer.
   
3. **Invoking the LLM:**
   - The LLM is invoked with all the provided information.
   - The `invoke_llm` function in the program is traced by LangSmith, recording all the runs which can be viewed on the LangSmith dashboard.

### Testset Generation

This project now includes functionality for generating a test set using LlamaIndex, Ollama, and HuggingFaceEmbedding. The generated test set can be used for various NLP tasks, including testing and evaluating models.

#### Testset Generation Process

1. **Loading Documents:**
   - Documents are loaded from the `./data` directory using `SimpleDirectoryReader`.
   
2. **Initializing Generator and Critic Models:**
   - Two instances of the Ollama model (`phi3:latest`) are initialized for generating and critiquing the test set.
   
3. **Initializing Embeddings:**
   - The script uses `HuggingFaceEmbedding` with the model `BAAI/bge-small-en-v1.5` for generating embeddings.
   
4. **Initializing Testset Generator:**
   - A `TestsetGenerator` is created using the generator and critic models, along with the embeddings.
   
5. **Generating Testset:**
   - The test set is generated from the documents. Note that the current configuration may have issues with the loop getting stuck at certain percentages. The process can take upwards of 5 minutes for one iteration, so please be patient.
   
6. **Writing Testset to CSV:**
   - The generated test set is saved as a CSV file named `testset.csv`.

### TruLens Integration

#### Adding Feedback Functions

1. **Initialize the Embedding Model:**
   - Uses `HuggingFaceEmbedding` with the model `BAAI/bge-small-en-v1.5`.

2. **Load Documents and Create Vector Store Index:**
   - Documents are loaded from the `./data` directory.
   - A vector store index is created from the documents.

3. **Initialize the LLM and Create a Query Engine:**
   - Uses the `Ollama` model (`phi3:latest`).

4. **Configure LiteLLM Provider for Feedback Functions:**
   - Sets up `LiteLLM` with the necessary configurations.

5. **Select Context for the Application:**
   - The context for the application is selected using `App.select_context`.

6. **Define Feedback Functions:**
   - Groundedness feedback function.
   - Relevance feedback functions for both answer and context.

7. **Initialize TruLlama Query Engine Recorder:**
   - Sets up `TruLlama` with the query engine and feedback functions.

8. **Query the Engine and Record the Process:**
   - Queries the engine with test questions and records the process.
   - Retrieves and prints feedback results.

#### Viewing Results and Running Dashboard

1. **Retrieve Feedback Results:**
   - Retrieves and prints feedback results from the recorded process.

2. **Access Records and Feedback:**
   - Access records and feedback using `Tru`.

3. **Run the Tru Dashboard:**
   - Uncomment the necessary line to run the Tru dashboard for visualizing the results.

## Future Work
1. Improve the method for providing context to the LLM.
2. Format the output from the model for better readability and usability.

## Notes
The following packages were added to run the Mixtral model, but the output was not convincing enough to use it:
```toml
sentencepiece = "^0.2.0"
protobuf = "^5.27.0"
```

## Problems Faced

### Poetry Install Stuck at Pending State
The installation process was stuck at the pending state. The issue was resolved by running the following command:
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
```
This solution was found by following the instructions mentioned in [this GitHub issue](https://github.com/python-poetry/poetry/issues/7235).

---

