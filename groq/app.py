## streamlit and os
import streamlit as st 
import os

## time to analyse how much it takes to give response
import time

## loading wepage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

## prompts
from langchain_core.prompts import ChatPromptTemplate

## groq
from langchain_groq import ChatGroq

## environment variable
from dotenv import load_dotenv

load_dotenv()

## load groq api key
groq_api_key=os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://langchain-ai.github.io/langgraph/tutorials/introduction/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectordb=FAISS.from_documents(st.session_state.final_docs,OllamaEmbeddings())
    
    
## title of streamlit
st.title("Groq demo")

## loading froq model
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model="Gemma2-9b-It"
    )

## prompt
prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions : {input}
    """
)

## document chain
document_chain=create_stuff_documents_chain(llm,prompt)
retriver=st.session_state.vectordb.as_retriever()
retriver_chain=create_retrieval_chain(retriever=retriver,combine_docs_chain=document_chain)


## taking input in streamlit
prompt=st.text_input("Input your prompt here")

## is prompt is given interact wit retrieval chain
if prompt:
    start=time.process_time()
    response=retriver_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    
    ## with a streamlit expander
    with st.expander("Document Similarity Search"):
        ## find the relevant chunks
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------")