# JSTBot: Advanced News Research Instrument
JSTBot stands at the forefront of news research tools, offering a seamless experience in extracting pertinent insights from the realms of the stock market and finance. This sophisticated platform accepts article URLs and poses questions, yielding relevant data with efficiency and precision.

# 🌟 Key Features
URL Integration: Directly input URLs or introduce text files brimming with URLs to access article contents.
Unstructured URL Processing: Engages LangChain's UnstructuredURL Loader for an in-depth article content analysis.
Enhanced Embedding Vector Construction: Employs OpenAI's state-of-the-art embeddings, coupled with the robust similarity search library, FAISS, to facilitate rapid and accurate information retrieval.
Interactive Querying: Pose queries and receive comprehensive responses from the LLM (ChatGPT), complete with source URLs for reference and verification.

# 🛠 Installation Guide
Repository Cloning: Execute the following command to clone the repository to your local system:
bash

# Copy code
git clone https://github.com/MGJillaniMughal/JSTBot-Advanced-News-Research-Instrument.git
Directory Navigation: Proceed to the project directory with:
bash

cd JSTBot-Advanced-News-Research-Instrument
Dependency Installation: Install the necessary dependencies using pip:

pip install -r requirements.txt
API Key Configuration: Set up your OpenAI API key by generating a .env file in the project's root directory and including your API key:
makefile
OPENAI_API_KEY=your_api_key_here

# 🚀 Usage Guidelines
Streamlit Application Initialization: Launch the Streamlit app using the command:
streamlit run main.py

# Web Application Interaction: The application will be accessible via your web browser.
URLs can be directly entered through the sidebar option.
Initiate comprehensive data processing with the "Process URLs" feature.
Witness the intricate procedures of text segmentation, embedding vector generation, and their subsequent indexing via FAISS.
The generated embeddings undergo storage and indexing through FAISS, ensuring an expedited retrieval process.
FAISS indices are preserved locally in a pickle format, guaranteeing their availability for subsequent utilization.
Engage with the system by posing inquiries and receiving detailed responses based on the analyzed news articles.
# 📂 Project Architecture
main.py: Primary script powering the Streamlit application.

requirements.txt: Enumerates the requisite Python packages.
faiss_store_openai.pkl: A designated pickle file for FAISS index storage.
.env: A configuration vessel for your OpenAI API key storage.
