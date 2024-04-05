import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import logging
from langchain_community.document_transformers import (
    LongContextReorder
)


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Set API key and endpoint as environment variables
os.environ["AZURE_OPENAI_API_KEY"] = "5623556c070842459be5cb014902b914"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://llm-aip-svc-dev.openai.azure.com/"

llm = AzureChatOpenAI(openai_api_version="2023-05-15", azure_deployment="gpt35",
                      temperature = 0.1,  
                      model_kwargs = {"top_p" : 0})


@st.cache_resource 
def load_and_process_documents(directory_path):
    # Load documents from a directory
    loader = DirectoryLoader(directory_path, show_progress=True)
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2800, chunk_overlap=200, length_function=len, is_separator_regex=False)
    docs = text_splitter.split_documents(docs)
    return docs

@st.cache_resource 
def setup_knowledge_base(_docs):
    # Set up embeddings
    embeddings = AzureOpenAIEmbeddings(azure_deployment="strong-emb", openai_api_version="2023-05-15")

    # Create a Chroma vector store from documents
    db = Chroma.from_documents(_docs, embeddings)
    return db

def setup_retriever(db):
    # Set up the MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
    return retriever

def generate_answer(question, llm, retriever):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query=question)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)

    # Prepare the prompt template
    QA_PROMPT = PromptTemplate(
        input_variables=["query", "contexts"], 
        template="""You are a helpful assistant who answers user queries using the
        contexts provided.

        Contexts:
        {contexts}

        Question: {query}""",
    )

    # Generate the answer
    qa_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    out = qa_chain(inputs={"query": question, "contexts": reordered_docs})
    
    # Return both the answer and the relevant documents
    return out["text"], reordered_docs

# Streamlit page configuration
st.set_page_config(page_title="Q&A Chatbot", layout="wide")

def main():
    st.title("Q&A Chatbot")
    
    if 'submitted_urn' not in st.session_state:
        st.session_state['submitted_urn'] = None
    if 'load_docs' not in st.session_state:
        st.session_state['load_docs'] = False
    
    if not st.session_state['load_docs']:
        urn_input = st.text_input("Enter the unique identifier (URN) to start:", key="new_urn")
        
        if st.button("Submit URN"):
            st.session_state['submitted_urn'] = urn_input
            st.session_state['load_docs'] = True

    if st.session_state['load_docs'] and st.session_state['submitted_urn']:
        directory_path = os.path.join(r"D:\Users\georgios.halios\Desktop\llm-experiment\CaseExplorer", st.session_state['submitted_urn'])
        docs = load_and_process_documents(directory_path)
        db = setup_knowledge_base(docs)
        retriever = setup_retriever(db)
        
        st.session_state["retriever"] = retriever

    if st.session_state['submitted_urn'] and 'retriever' in st.session_state:
        question = st.text_input("Ask me anything:", key="question")

        if st.button("Get Answer", key="get_answer"):
            answer, relevant_docs = generate_answer(question, llm, st.session_state.retriever)
            st.write("Answer:", answer)

            st.subheader("Relevant Chunks/Docs")
            # Use Streamlit's expander for each chunk
            for index, doc in enumerate(relevant_docs, start=1):
                with st.expander(f"Chunk {index}", expanded=False):
                    st.markdown(f"<div style='line-height: 1.15;'>{doc.page_content}</div>", unsafe_allow_html=True)
                    
                    # Display metadata if available
                    if doc.metadata:
                        st.markdown(f"<small>Metadata: {doc.metadata}</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()