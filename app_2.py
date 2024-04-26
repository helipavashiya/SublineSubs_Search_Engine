import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="my_chromadb_1")

# distilbert-base-nli-mean-tokens model for embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="distilbert-base-nli-mean-tokens")

# get or create the collection
collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})

def main():
    st.title("✨SublimeSubs Search✨")

    # getting the user input
    user_query = st.text_input("Enter Your Query")

    # Create a container for buttons
    button_container = st.container()

    # Add Search and Clear buttons to the container
    with button_container:
        search_clicked = st.button("Search")
        st.snow()
        st.write("")  # Add space between buttons
        clear_clicked = st.button("Clear")

    if search_clicked:
        if user_query:
            # query the collection
            results = collection.query(
                query_texts=[user_query],
                n_results=50,  # Retrieve more results to ensure enough unique documents are found
                include=['documents', 'distances', 'metadatas']
            )

            # collect all documents
            all_documents = results['documents'][0]

            # select top 10 unique documents
            unique_documents = []
            for document in all_documents:
                if document not in unique_documents:
                    unique_documents.append(document)
                if len(unique_documents) == 10:
                    break

            # display user input
            st.write(f"Your search query: {user_query}")

            # display unique output documents
            st.write("Search Results:")
            for i, document in enumerate(unique_documents, 1):
                st.write(f"{i}. {document}")

    elif clear_clicked:
        user_query = " "  # Reset the user query input

if __name__ == "__main__":
    main()

