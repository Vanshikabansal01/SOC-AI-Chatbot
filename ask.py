from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)

retriever = vector_db.as_retriever(search_kwargs={"k":2})

llm = Ollama(model="tinyllama")

while True:
    query = input("\nSOC Question: ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    if not docs:
        print("No relevant documentation found.")
        continue

    context = "\n".join([doc.page_content[:500] for doc in docs])

    prompt = f"""
You are an expert cybersecurity SOC analyst.

Use the provided documentation context to answer the question.

If the question relates to a security incident, include:

1. Incident description
2. Possible cause
3. Investigation steps
4. Containment actions
5. Remediation steps

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n", response)

    print("\nSources:")
    for doc in docs:
        print(doc.metadata.get("source"))