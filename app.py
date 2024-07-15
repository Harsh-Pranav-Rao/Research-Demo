import streamlit as st
import pandas as pd
import uuid
from haystack.telemetry import tutorial_running
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline

# Initialize telemetry
tutorial_running(27)

# Set up document store
document_store = InMemoryDocumentStore()

# Load custom dataset
@st.cache
def load_custom_dataset(file_path):
    df = pd.read_csv(file_path)
    # Ensure 'abstract' and 'title' columns are present and not null
    df = df.dropna(subset=['abstract', 'title'])
    
    # Remove duplicates based on 'abstract' column to ensure uniqueness
    df = df.drop_duplicates(subset=['abstract'])
    
    # Generate unique IDs and check for duplicates
    unique_ids = set()
    docs = []
    for idx, row in df.iterrows():
        doc_id = str(uuid.uuid4())
        while doc_id in unique_ids:  # Ensure unique ID
            doc_id = str(uuid.uuid4())
        unique_ids.add(doc_id)
        docs.append(Document(content=row['abstract'], meta={'title': row['title']}, id=doc_id))
    
    return docs

def main():
    st.title("GPT-3.5 Turbo Q&A")

    uploaded_file = st.file_uploader("Upload your custom dataset (CSV format)", type="csv")

    if uploaded_file:
        docs = load_custom_dataset(uploaded_file)

        # Embed documents
        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        doc_embedder.warm_up()
        docs_with_embeddings = doc_embedder.run(docs)
        
        # Check for duplicate IDs in docs_with_embeddings
        unique_ids = set()
        unique_docs = []
        for doc in docs_with_embeddings["documents"]:
            if doc.id not in unique_ids:
                unique_ids.add(doc.id)
                unique_docs.append(doc)
        
        # Write unique documents to document store
        try:
            document_store.write_documents(unique_docs)
        except Exception as e:
            st.error(f"Error writing documents to document store: {str(e)}")
            st.stop()

        # Set up text embedder
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

        # Set up retriever
        retriever = InMemoryEmbeddingRetriever(document_store)

        # Set up prompt template
        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
        prompt_builder = PromptBuilder(template=template)

        # Set up OpenAI generator
        generator = OpenAIGenerator(model="gpt-3.5-turbo")

        # Build pipeline
        basic_rag_pipeline = Pipeline()
        basic_rag_pipeline.add_component("text_embedder", text_embedder)
        basic_rag_pipeline.add_component("retriever", retriever)
        basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
        basic_rag_pipeline.add_component("llm", generator)
        basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
        basic_rag_pipeline.connect("prompt_builder", "llm")

        question = st.text_input("Ask a question based on the title to get the abstract:")
        if st.button("Get Answer"):
            if question:
                response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
                st.write(response["llm"]["replies"][0])
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
