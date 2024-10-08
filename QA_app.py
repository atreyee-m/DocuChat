import re
import os, shutil
import chainlit as cl
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

faiss_index = os.path.join(os.path.expanduser("~"), "Documents/side projects/documentQA/faiss_index")
data = os.path.join(os.path.expanduser("~"), "Library/Mobile Documents/com~apple~CloudDocs/side projects/data")

PROMPT_TEMPLATE = """
You are a machine learning expert assistant. Use the context below to answer the user's question.

Context:
{context}

Question: {question}

Answer:
"""

def create_llm():
    return Ollama(model="mistral")

def create_embedding_model():
    return OllamaEmbeddings(model="nomic-embed-text")

def load_pdfs(chunk_size=3000, chunk_overlap=100):
    loader = PyPDFDirectoryLoader(data)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents=documents)
    return docs



#  check if vector store exists
def vector_store_exists():
    index_file = os.path.join(faiss_index, 'index.faiss')
    return os.path.exists(index_file)


def create_vector_store(docs):
    embedding_model = create_embedding_model()


    try:
        # Create FAISS index from documents
        print(f"Creating FAISS index from {len(docs)} documents.")
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(faiss_index)  # Save FAISS index to the path
        print(f"FAISS index created and saved at {faiss_index}")
        return vector_store
    except Exception as e:
        print(f"Error during FAISS index creation: {e}")
        return None


# Fload the FAISS vector store, and create it if not present
def load_vector_store(docs):
    index_file = os.path.join(faiss_index, 'index.faiss')

    # Check if the FAISS index file exists
    if os.path.exists(index_file):
        embedding_model = create_embedding_model()
        print(f"Loading FAISS index from {faiss_index}")
        return FAISS.load_local(faiss_index, embedding_model, allow_dangerous_deserialization=True)
    else:
        print(f"FAISS index does not exist at {faiss_index}. Creating it now.")
        # Create the vector store if the index does not exist
        return create_vector_store(docs)

# Create the QA chain with memory and a custom prompt
def create_qa_chain(llm, vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 10}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
    return qa_chain

@cl.on_chat_start
async def start_chat():
    # Load Llama model and FAISS vector store
    llm = create_llm()

    # Load PDF documents and ensure the FAISS vector store is loaded or created
    docs = load_pdfs()
    vector_store = load_vector_store(docs)
    if vector_store is None:
        await cl.Message(content="Error: Unable to create or load FAISS vector store.").send()
        return
    # Create the QA chain
    qa_chain = create_qa_chain(llm, vector_store)

    # Add custom messages to the user interface
    msg = cl.Message(content="Loading the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the QA Chatbot! Please ask your question."
    await msg.update()

    # Store the QA chain in the user session
    cl.user_session.set('qa_chain', qa_chain)
    cl.user_session.set('question_count', 0)  # Initialize question count




@cl.on_message
async def generate_response(query):
    # Retrieve QA chain and question count from the user session
    qa_chain = cl.user_session.get('qa_chain')

    processing_msg = cl.Message(content="Processing your question, please wait...")
    await processing_msg.send()

    # Generate the response using the QA chain
    res = await qa_chain.acall(query.content, callbacks=[cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
    )])

    # Extract and send the result
    result, source_documents = res['answer'], res['source_documents']

    # Extract all values associated with the 'metadata' key
    metadata_values = re.findall(r"metadata={'source': '([^']*)', 'page': (\d+)}", str(source_documents))
    metadata_string = "\n".join([f"Source: {source}, page: {page}" for source, page in metadata_values])

    # Append sources to the result
    result += f'\n\nSources:\n{metadata_string}'
    await cl.Message(content=result).send()
