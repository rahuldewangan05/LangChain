## for ui
import streamlit as st
import os

## document loader
from langchain_community.document_loaders import PyPDFDirectoryLoader

## text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

## vector_Store
from langchain_objectbox.vectorstores import ObjectBox

## for using open-source llm "GROQ"
from langchain_groq import ChatGroq

## to make prompts to send to llm
from langchain_core.prompts import ChatPromptTemplate

## stuff chain to interact with llm
from langchain.chains.combine_documents import create_stuff_documents_chain

## retrival chain to fetch relevent document form vector store
from langchain.chains import create_retrieval_chain

## for loading environment variable
from dotenv import load_dotenv

## loading environment variable
load_dotenv()

## load hugging face emedding and groq for llama3
groq_Api_key=os.getenv("GROQ_API_KEY")
huggingface_embedding=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

## title of ui
st.title("ObjectBox VectorStore Demo")

## loading llm
llm=ChatGroq(groq_api_key=groq_Api_key,model_name="Llama3-8b-8192")

## prompt
prompt=ChatPromptTemplate.from_template(
    """ 
    Answer the question based on the povided context only.
    Please provide the most accurate response based on the question.
    <context>   
    {context}
    <context>
    Questions:{input}
    """
)

## vector embedding and vectorStore
def vector_embeddings():
    if "vectordb" not in st.session_state:
        st.session_state.loaders=PyPDFDirectoryLoader("./us-census")
        st.session_state.docs=st.session_state.loaders.load()
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.documnet=st.session_state.splitter.split_documents(documents=st.session_state.docs)
        st.session_state.vectodb=ObjectBox.from_documents(documents=st.session_state.documnet, embedding=huggingface_embedding)



## taking input from user
input_prompt=st.text_input("Enter your question from Document")

## button to call vector embedding function to load document in vector store
if st.button("Document Embedding"):
    vector_embeddings()
    st.write("Document loaded in ObjectBox VerctorStore")
    
import time

## if the input is given calculating time taken by model to respond
if input_prompt:
    ## making stuff chain and retriever chain when getting input prompt
    document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
    retriver=st.session_state.vectordb.as_retriever()
    retriver_chain=create_retrieval_chain(retriver,document_chain)
    
    ## noting time taken to get response
    start=time.process_time()
    output=retriver_chain.invoke({"input":input_prompt})
    print("Response time : ",time.process_time()-start)
    
    st.write(output["answer"])
    
    ## also displaying the context (relevent document for that response)
    with st.expander("Document Similarity Search"):
        ## find relevant document
        for i,doc in enumerate(output["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------------------------")