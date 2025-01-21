
---

# Contextual RAG App

A **Contextual RAG App** built with Streamlit that uses the Retrieval-Augmented Generation (RAG) approach to answer questions based on a given context. This app integrates **LangChain**, **OpenAI**, **ChromaDB**, and **Athina AI** for efficient document retrieval, contextual compression, and response evaluation.

---

## Features

- **Document Loading and Indexing**: Automatically loads and indexes documents from a CSV file.
- **RAG Workflow**: Combines retrieval and generation to answer queries.
- **Evaluation**: Uses Athina AI for evaluating context relevancy.
- **Streamlit Interface**: User-friendly interface with text input and evaluation options.
- **Secure API Keys**: Handles sensitive information securely using `secrets.toml`.

---

## File Structure

```
contextual_rag_app/
├── app.py                    # Main Streamlit application
├── context.csv               # Data file for RAG system
├── requirements.txt          # Python dependencies
└── .streamlit/
    └── secrets.toml          # Secure storage for API keys
```

---

## Installation

### Prerequisites

- Python 3.8+
- API keys for OpenAI and Athina AI

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/contextual_rag_app.git
   cd contextual_rag_app
   ```

2. Create the `context.csv` file:
   - Format the file as required by the **LangChain CSVLoader** (e.g., columns like `content` and `metadata`).

3. Add your API keys to `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key"
   ATHINA_API_KEY = "your_athina_api_key"
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser (usually at `http://localhost:8501`).

3. Interact with the app:
   - Enter a query in the text input field.
   - View the generated response and the retrieved context.
   - Click "Run Evaluation" to evaluate the response with Athina AI.

---

## Technologies Used

- **Streamlit**: Interactive web application framework.
- **LangChain**: For RAG and document retrieval.
- **OpenAI**: Language model for response generation.
- **ChromaDB**: Vector store for efficient document retrieval.
- **Athina AI**: Evaluation of context relevancy.
- **Pandas**: Data manipulation and evaluation results display.

---

## Key Files

1. **`app.py`**:
   - Main application code for the Streamlit app.
   - Includes indexing, retrieval, and RAG workflow setup.

2. **`context.csv`**:
   - The input file containing documents to be indexed.

3. **`requirements.txt`**:
   - List of Python libraries and versions required to run the app.

4. **`.streamlit/secrets.toml`**:
   - Securely stores API keys. *Do not include this file in your public repository*.

---

## How It Works

1. **Document Indexing**:
   - Documents from `context.csv` are split and stored as embeddings in ChromaDB.

2. **Retrieval-Augmented Generation (RAG)**:
   - Retrieves the most relevant documents for a query.
   - Uses OpenAI's language model to generate a response based on the retrieved context.

3. **Evaluation**:
   - Prepares data for evaluation.
   - Uses Athina AI to assess the relevancy of the retrieved context.

---

## Example Output

1. **User Query**: "What is the capital of France?"
2. **Generated Response**: "The capital of France is Paris."
3. **Retrieved Context**: 
   - Relevant documents from `context.csv` are displayed.
4. **Evaluation Results**:
   - Context relevancy scores displayed in a table.

---

## Notes

- **Security**: Ensure `.streamlit/secrets.toml` is included in `.gitignore` to prevent exposing API keys.
- **Costs**: Using OpenAI and Athina AI may incur charges. Monitor API usage carefully.
- **Error Handling**: Add robust error handling for production-level deployments.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## Contact

For questions or feedback, feel free to reach out:

- **GitHub**: [YourUsername](https://github.com/your-username)
- **Email**: your-email@example.com

---

Let me know if you'd like to personalize this further!
