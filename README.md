# DocuChat

DocuChat is an document-based Question Answering (QA) chatbot built using LangChain and FAISS. It leverages the Ollama model for natural language processing and document embedding. The chatbot processes PDFs, builds a FAISS index, and allows users to ask questions while retrieving relevant context and source information from the documents.

Features
- Conversational Memory: Maintains context throughout the conversation.
- Document Retrieval: Uses FAISS for efficient similarity search across documents.
 PDF Support: Loads and processes PDF documents from a specified directory.
- Source Tracking: Provides the sources and page numbers of the document references used to answer questions.
- Extensible: Built with LangChain, making it flexible for integration with different models and document loaders.
