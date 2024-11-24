import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (e.g., API keys)
load_dotenv()

st.title("Huduma Chatbot")

# Load the document
loader = PyPDFLoader("FINAL PROJECT REPORT huduma center.pdf")
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create embeddings using GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store the document embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up a retriever for similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize the Google Gemini model for the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Input query from user
query = st.text_input("Ask me Anything: ") 
prompt = query

# Define system prompt for the chatbot
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Run the RAG (Retrieval-Augmented Generation) process if query is provided
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke the RAG chain and get the response
    response = rag_chain.invoke({"input": query})

    # Display the answer in Streamlit
    st.write(response["answer"])