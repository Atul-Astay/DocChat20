import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Api configuration
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash',
                             temperature = 0.7)

loader = PyPDFLoader("C:\\Users\\dango\\Downloads\\AI_ML_Engineer_Interview_QA.pdf")
doc = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                          chunk_overlap = 40)

split_doc = splitter.split_documents(doc)

google_embd = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = FAISS.from_documents(documents=split_doc,
                                   embedding = google_embd)


retriever = vectorstore.as_retriever(search_type = 'similarity')

st.title("Chat with PDF")

query = st.chat_input("Pass your prompt here")

system_output = (
    "You are my personal assistant to talk with PDF"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [('system',system_output),
     ("human","{input}")]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever,
                                       question_answer_chain)

    response = rag_chain.invoke(({'input':query}))

    st.write("User Query :",query)

    print(response['answer'])

    st.write("Chatbot Response :",response['answer'])