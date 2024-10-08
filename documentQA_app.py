from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
# from langchain.llms.bedrock import Bedrock

# from langchain.llms.bedrock import Bedrock
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

faiss_index = "/Users/atreyeemukherjee/Documents/side projects/documentQA/faiss_index"
data = "/Users/atreyeemukherjee/Library/Mobile Documents/com~apple~CloudDocs/side projects/data"

def load_pdfs(chunk_size=3000, chunk_overlap=100):

    # load the pdf documents
    loader=PyPDFDirectoryLoader(data)
    documents=loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents=documents)
    return docs

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(docs):
    # Use Ollama embeddings instead of Bedrock
    ollama_embeddings = OllamaEmbeddings(model="llama3")  # Specify your Ollama model

    # Create and save the vector store using FAISS
    vector_store = FAISS.from_documents(docs, ollama_embeddings)
    vector_store.save_local("/Users/atreyeemukherjee/Documents/side projects/documentQA/faiss_index")
    
    return None

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama  # This is the LLM for generating responses
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

def create_llm():
    llm = Ollama(model="llama3")
    return llm

llama3_embeddings = OllamaEmbeddings(model="llama3")
vector_store = FAISS.load_local(faiss_index, llama3_embeddings, allow_dangerous_deserialization=True)

# Create memory history for the conversation
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)

# Create the QA chain with ConversationalRetrievalChain
llm = create_llm()  # Use the LLM instead of the embeddings here
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    chain_type='stuff', 
    retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 3}),
    return_source_documents=True,
    memory=memory
)

import chainlit as cl
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

@cl.on_chat_start
async def create_qa_chain():

    # Load Llama 3 model for the LLM
    llm = Ollama(model="llama3")  # Load Llama 3 as the LLM

    # Load embeddings and vector store with Llama 3 embeddings
    llama3_embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.load_local('faiss_index', llama3_embeddings, allow_dangerous_deserialization=True)
    
    # Create memory history for conversation context
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create the QA chain with ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff', 
        retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory
    )
    
    # Add custom messages to the user interface
    msg = cl.Message(content="Loading the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the QA Chatbot! Please ask your question."
    await msg.update()
    
    # Store the QA chain in the user session
    cl.user_session.set('qa_chain', qa_chain)

import re
import chainlit as cl

@cl.on_message
async def generate_response(query):
    qa_chain = cl.user_session.get('qa_chain')

    res = await qa_chain.acall(query.content, callbacks=[cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, 
        )])

    # extract results and source documents
    result, source_documents = res['answer'], res['source_documents']

    # Extract all values associated with the 'metadata' key
    source_documents = str(source_documents)
    metadata_values = re.findall(r"metadata={'source': '([^']*)', 'page': (\d+)}", source_documents)

    # Convert metadata_values into a single string
    pattern = r'PDF Documents|\\'
    metadata_string = "\n".join([f"Source: {re.sub(pattern, '', source)}, page: {page}" for source, page in metadata_values])

    # add metadata (i.e., sources) to the results
    result += f'\n\n{metadata_string}'

    # send the generated response to the user
    await cl.Message(content=result).send()