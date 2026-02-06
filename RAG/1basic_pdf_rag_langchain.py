# basic_pdf_rag_langchain.py
# pip install langchain langchain-community langchain-ollama pypdf faiss-cpu

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys
import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"   # ← try this first 

if len(sys.argv) != 2:
    print("Usage: python basic_pdf_rag_langchain.py your_paper.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]

# 1. Load & split PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(f"Loaded {len(docs)} pages → {len(chunks)} chunks")



# 2. Embed + store (in-memory FAISS)
embed_model = OllamaEmbeddings(model="nomic-embed-text")   # or "mxbai-embed-large" etc.
vectorstore = FAISS.from_documents(chunks, embed_model)
#retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# 3. LLM
llm = ChatOllama(model="llama3.2", temperature=0.1)   # or mistral, phi3, gemma2 etc.

# 4. Very simple prompt
prompt = ChatPromptTemplate.from_template(
    """Answer using only the context. If unsure, say you don't know.

Context: {context}

Question: {question}

Answer:"""
)

# 5. Chain
chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nPDF RAG ready. Type your questions (or 'exit'):\n")

while True:
    q = input("> ").strip()
    if q.lower() in ["exit", "quit", "q"]:
        break
    if not q:
        continue
    try:
        answer = chain.invoke(q)
        print("\n" + answer + "\n")
    except Exception as e:
        print(f"Error: {e}")
