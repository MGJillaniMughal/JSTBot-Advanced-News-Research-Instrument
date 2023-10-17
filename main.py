import os
import streamlit as st
import pickle
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Function to load data from URLs
def load_article_data(urls):
    """
    Load and return article data from the specified URLs.

    :param urls: List of article URLs.
    :return: Loaded article data.
    """
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        return data
    except Exception as e:
        st.error(f"Error loading article data: {e}")
        return None

# Function to process article data
def process_article_data(data):
    """
    Process the article data: split text, generate embeddings, and save to FAISS index.

    :param data: Raw article data.
    :return: FAISS index object.
    """
    try:
        # Split text into manageable parts
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\\n", "\\r", "\\t", ".", "?", "!", "\\n\\n", ","],
            chunk_size=1000,
        )
        docs = text_splitter.split_documents(data)

        # Generate embeddings for the split text
        embedding = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embedding)

        return vector_store
    except Exception as e:
        st.error(f"Error processing article data: {e}")
        return None

# Main execution starts here
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Set up Streamlit
    st.title("JSTBot: Your Advanced News Research Companion 🚀")
    st.sidebar.title("Configuration")

    # Collect URLs from the user
    urls = [st.sidebar.text_input(f"URL {i+1}", value="") for i in range(3) if st.sidebar.text_input(f"URL {i+1}", value="")]

    # Trigger processing
    if st.sidebar.button("Process URLs 🚀") and urls:
        with st.spinner("Loading and processing articles... 🔄"):
            article_data = load_article_data(urls)
            if article_data is not None:
                faiss_index = process_article_data(article_data)
                
                if faiss_index is not None:
                    # Save the FAISS index for future use
                    with open("faiss_index.pkl", "wb") as f:
                        pickle.dump(faiss_index, f)
                    st.success("Articles processed and indexed successfully! ✅")
                else:
                    st.error("Failed to generate FAISS index ❌")
            else:
                st.error("Failed to load article data ❌")

    # Querying the system
    query = st.text_input("Enter your query here: 🕵️")
    if query:
        if os.path.exists("faiss_index.pkl"):
            with st.spinner("Retrieving information... 🧠"):
                try:
                    with open("faiss_index.pkl", "rb") as f:
                        faiss_index = pickle.load(f)

                    # Initialize the LLM
                    llm = OpenAI(temperature=0.9, max_tokens=500, top_p=1, frequency_penalty=0, presence_penalty=0.6)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss_index.as_retriever())
                    result = chain({'question': query}, return_only_outputs=True)

                    # Display the result
                    st.write(result)
                except Exception as e:
                    st.error(f"Error retrieving information: {e}")
        else:
            st.error("No indexed data available for querying ❌")
