import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="Talk to Penelope",
                   page_icon="🕊️")  # Page Config


def format_docs(docs):  # Cleans the text for the bot to read
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        # Takes the Penelope narrator tag
        narrator = doc.metadata.get("narrator", "Penelope (General)")
        formatted_chunks.append(
            f"SOURCE: {source} | POV: {narrator}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


# The temperature is kept at 0.0 to make sure the model doesn't diverge from Penelope's life
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)


@st.cache_resource
def initialize_penelope():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)

    # Loads the provided data, which is treated as "data"
    data_loader = TextLoader("Library/penelope_data.txt", encoding='utf-8')
    data_raw = data_loader.load()
    data_chunks = splitter.split_documents(data_raw)
    for d in data_chunks:
        d.metadata["source"] = "penelope_data.txt"

    data_store = FAISS.from_documents(data_chunks, embeddings)

    # Loads the book, which is treated as "memories"
    book_loader = TextLoader(
        "Library/ScatteredSoul.txt", encoding='utf-8')
    book_raw = book_loader.load()
    book_chunks = splitter.split_documents(book_raw)
    for d in book_chunks:
        d.metadata["source"] = "ScatteredSoul.txt"

    memory_store = FAISS.from_documents(book_chunks, embeddings)

    return {
        # Data gets the least examples but still gets 2 relevant chunks
        "data": data_store.as_retriever(search_kwargs={"k": 2}),
        # Memories has the most examples since it's the book itself
        "memories": memory_store.as_retriever(search_kwargs={"k": 8}),
    }


retriever = initialize_penelope()

system_prompt_str = (
    "You are Penelope from the book Blue and the Scattered Soul."
    "Context:\n{context}"
)


def build_combined_context(input_text):  # Builds the AI's context

    data_chunks = retriever["data"].invoke(input_text)  # Data
    memory_chunks = retriever["memories"].invoke(
        input_text)  # Scattered Soul

    context_str = "CRITICAL DATA FACTS:\n"
    context_str += format_docs(data_chunks)
    context_str += "\n\nLIFE MEMORIES:\n"
    context_str += format_docs(memory_chunks)
    return context_str


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_str),
    ("human", "{input}"),
])


rag_chain = (
    {
        "context": RunnableLambda(build_combined_context),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

st.title("Penelope")
question = st.text_input("Talk to Penelope.")

if question:
    response = rag_chain.invoke(question)
    st.write(response)
