DocuChat is a document-based Question Answering (QA) chatbot built using a Retrieval-Augmented Generation (RAG) architecture. It leverages LangChain and FAISS for efficient document retrieval and the Ollama model for natural language processing and document embedding. The chatbot processes PDFs, builds a FAISS index, and allows users to ask questions while retrieving relevant context and source information from the documents to generate accurate answers.

Features
- Conversational Memory: Maintains context throughout the conversation.
- Document Retrieval: Uses FAISS for efficient similarity search across documents.
- PDF Support: Loads and processes PDF documents from a specified directory.
- Source Tracking: Provides the sources and page numbers of the document references used to answer questions.
- Extensible: Built with LangChain, FAISS, Ollama, and Chainlit, providing flexibility for integration with different models, document loaders, and vector search systems.
