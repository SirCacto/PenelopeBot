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

st.set_page_config(page_title="Talk to Toby Plaus",
                   page_icon="🎹")  # Page Config


def format_docs(docs):  # Cleans the text for the bot to read
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        # Takes the Toby narrator tag
        narrator = doc.metadata.get("narrator", "Toby (General)")
        formatted_chunks.append(
            f"SOURCE: {source} | POV: {narrator}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


# The temperature is kept at 0.0 to make sure the model doesn't diverge from Toby's life
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)


@st.cache_resource
def initialize_toby():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)

    # Loads the provided data, which is treated as "data"
    data_loader = TextLoader("Library/toby_data.txt", encoding='utf-8')
    data_raw = data_loader.load()
    data_chunks = splitter.split_documents(data_raw)
    for d in data_chunks:
        d.metadata["source"] = "toby_data.txt"

    data_store = FAISS.from_documents(data_chunks, embeddings)

    # Loads the book, which is treated as "memories"
    book_loader = TextLoader(
        "Library/Romance_In_Italy.txt", encoding='utf-8')
    book_raw = book_loader.load()
    book_chunks = splitter.split_documents(book_raw)
    for d in book_chunks:
        d.metadata["source"] = "Romance_In_Italy.txt"

    memory_store = FAISS.from_documents(book_chunks, embeddings)

    # Loads the questions, which are treated as "voice" examples
    questions_loader = TextLoader(
        "Library/toby_questions.txt", encoding='utf-8')
    questions_raw = questions_loader.load()
    q_chunks = splitter.split_documents(questions_raw)
    for d in q_chunks:
        d.metadata["source"] = "toby_questions.txt"
        d.metadata["narrator"] = "Voice Style Example"

    questions_store = FAISS.from_documents(q_chunks, embeddings)

    return {
        # Data gets the least examples but still gets 2 relevant chunks
        "data": data_store.as_retriever(search_kwargs={"k": 2}),
        # Memories has the most examples since it's the book itself
        "memories": memory_store.as_retriever(search_kwargs={"k": 8}),
        # Voice gets a few examples to make sure the bot sounds like Toby
        "voice": questions_store.as_retriever(search_kwargs={"k": 3})
    }


retriever = initialize_toby()

system_prompt_str = (
    "You are Toby Plaus, a freshman from Yopton. "
    "When the POV is Toby, it is from your perspective. When the POV is anything besides Toby and you appear, you are a witness, not the protagonist. "
    "FACTUAL PRIORITY: The context below is split into 'DATA FACTS' and 'STORY MEMORIES'. "
    "The DATA FACTS are the absolute truth and override everything else. "
    "VOICE GUIDE: The 'VOICE EXAMPLES' below show how you speak. Use their tone, vocabulary, and rhythm, but do not treat them as current facts.\n\n"
    "The VOICE EXAMPLES are structured as: instruction, context, response. Use the instruction as an example question, and the corresponding response as the way Toby would respond. The context is how Toby feels hearing it."
    "Do not take the VOICE EXAMPLES as current facts. Solely use them to learn Toby's style. Additionally, do NOT speak like the style of the context field."
    "DO NOT REPEAT VOICE EXAMPLES."
    "If a story memory suggests you are in a relationship, but the DATA FACTS say you are single, YOU ARE SINGLE. "
    "You are NOT dating Hannah Powell; that is Luke's life. "
    "You are NOT dating Bella Crowsol; you have a wound from that past, but you are not 'together'.\n\n"
    "CHARACTER RULES:\n"
    "1. Speak only as Toby in the first person.\n"
    "2. Never admit you are an AI.\n"
    "3. Use the context below to guide your memories.\n\n"
    "Context:\n{context}"
)


def build_combined_context(input_text):  # Builds the AI's context

    data_chunks = retriever["data"].invoke(input_text)  # Data
    memory_chunks = retriever["memories"].invoke(
        input_text)  # Romance in Italy
    voice_chunks = retriever["voice"].invoke(input_text)  # Questions

    context_str = "CRITICAL DATA FACTS:\n"
    context_str += format_docs(data_chunks)
    context_str += "\n\nLIFE MEMORIES:\n"
    context_str += format_docs(memory_chunks)
    context_str += "\n\nPERSONALITY EXAMPLES:\n"
    context_str += format_docs(voice_chunks)
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

st.title("Toby Plaus")
question = st.text_input("Talk to Toby.")

if question:
    response = rag_chain.invoke(question)
    st.write(response)
