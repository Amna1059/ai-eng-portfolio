import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")
    return loader.load()

def build_rag_pipeline(file_path):
    print(f"Loading document: {file_path}")
    documents = load_document(file_path)
    print(f"Loaded {len(documents)} page(s)/section(s)")

    full_text = "\n\n".join([doc.page_content for doc in documents])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    return retriever, llm, full_text

def ask_question(question, retriever, llm, full_text):
    summary_keywords = ["summarize", "summary", "how many questions", "complete document", "whole document", "entire document", "all questions"]

    if any(kw in question.lower() for kw in summary_keywords):
        context = full_text
        docs = []
    else:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based only on the following context.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, docs

if __name__ == "__main__":
    file_path = input("Enter the path to your document (PDF/DOCX/TXT): ").strip()
    retriever, llm, full_text = build_rag_pipeline(file_path)

    print("\nDocument ready! Ask questions (type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nYour question: ")
        if question.lower() == "quit":
            break

        answer, sources = ask_question(question, retriever, llm, full_text)
        print(f"\nAnswer: {answer}")
        print(f"\n(Based on {len(sources)} source chunks)" if sources else "\n(Used full document)")