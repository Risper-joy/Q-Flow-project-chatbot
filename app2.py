from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load the document
loader = PyPDFLoader("FINAL PROJECT REPORT huduma center.pdf")
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create embeddings and vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize Google Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Define RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('text')
    if not user_input:
        return jsonify({"response": "Invalid input."})
    
    # Process query through the QA chain
    response = qa_chain.run(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
