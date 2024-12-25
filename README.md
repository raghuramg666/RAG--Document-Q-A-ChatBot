# RAG Document Q&A ChatBot

## Overview
The RAG Document Q&A ChatBot is an interactive chatbot application that leverages the Retrieval-Augmented Generation (RAG) architecture to provide accurate and context-aware answers to user queries based on a reference document. This project is designed to demonstrate the integration of advanced AI techniques for document-based question answering.

---

## Features
- Document Understanding: Utilizes a PDF file as the source for generating accurate answers.
- Retrieval-Augmented Generation: Combines retrieval mechanisms with generative models for precise and contextual responses.
- Interactive Application: Easy-to-use interface for querying and receiving responses.

---

## Project Structure
- `chatbotapp.py`: The main application file, containing the logic for chatbot interactions and RAG implementation.
- `chroma_store/`: Directory for storing embeddings or intermediate files generated during processing.
- `data.pdf`: A reference document used for answering user queries.

---

## Requirements
To run this project, you will need the following:
- Python 3.8 or above
- Required Python libraries (listed in `requirements.txt`, if available)
- Libraries for handling PDFs and building AI models, such as `PyPDF2`, `LangChain`, and `ChromaDB`.

---

## Setup and Usage

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RAG--Document-Q-A-ChatBot-main
```

### 2. Install Dependencies
Install all required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Add Your Data
Replace or add your PDF file under the project directory, ensuring the file path matches the one in the application.

### 4. Run the Application
Start the chatbot application:
```bash
python chatbotapp.py
```

### 5. Query the ChatBot
Interact with the chatbot to ask questions related to the uploaded document.

---

## Future Enhancements
- Integration with cloud-based document storage.
- Adding support for multiple document formats (e.g., Word, Excel).
- Deployment on web platforms using frameworks like Flask or FastAPI.



