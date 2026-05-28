# EEG RAG Assistant

A question-answering tool built on top of a resting-state EEG anxiety classification paper. Ask it methodological questions, get answers grounded in the actual text.

Built to demonstrate how retrieval-augmented generation works in a real research context, using a paper I wrote as the knowledge base.

## What it does

You run two scripts. The first reads a PDF, breaks it into chunks, embeds them using a local sentence transformer model, and stores them in a local vector database. The second loads that database, takes a question as input, retrieves the most relevant chunks, and passes them to Claude to generate an answer.

## Stack

- LangChain — orchestration and retrieval
- ChromaDB — local vector store
- HuggingFace sentence-transformers — embeddings (all-MiniLM-L6-v2)
- Claude (Anthropic API) — answer generation
- pypdf — PDF parsing

## Setup

```bash
git clone https://github.com/joshthrelkeld/eeg-rag-assistant.git
cd eeg-rag-assistant
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-anthropic langchain-community langchain-chroma langchain-huggingface chromadb pypdf sentence-transformers python-dotenv
Add your Anthropic API key to a .env file:
```
ANTHROPIC_API_KEY=your_key_here

Drop a PDF into the `data/` folder and name it `eeg_paper.pdf`.

## Usage

```bash
python ingest.py   # embed the PDF and build the vector store
python query.py    # start the question-answering loop
## Example
```
Ask a question: What were the main findings for trait anxiety classification?
Answer: Trait anxiety classification dropped to 39-43% accuracy across all models, which is below chance. The authors interpret this not as a failure of the approach but as evidence that trait anxiety, as a dispositional construct, is not reliably detectable from a single two-minute resting-state session.

## Background

The paper this was built on classifies state and trait anxiety from resting-state EEG using spectral band power features and three classifiers (logistic regression, SVM, random forest). The full paper and code are available at [eeg_anxiety](https://github.com/joshthrelkeld/eeg_anxiety).
