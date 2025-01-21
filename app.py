import os
import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset

# Set secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ['ATHINA_API_KEY'] = st.secrets["ATHINA_API_KEY"]

# --- Indexing ---
@st.cache_resource
def load_and_index_data():
    # load embedding model
    embeddings = OpenAIEmbeddings()

    # load data
    loader = CSVLoader("./context.csv")
    documents = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    # create vectorstore
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore

vectorstore = load_and_index_data()


# --- Retriever ---
retriever = vectorstore.as_retriever()


# --- Contextual Retriever ---
llm = ChatOpenAI()
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


# --- RAG Chain ---
template = """
You are a helpful assistant that answers questions based on the following context.
If you don't find the answer in the context, just say that you don't know.
Context: {context}

Question: {input}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": compression_retriever,  "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- Streamlit App ---
st.title("Contextual RAG App")

query = st.text_input("Enter your question:")

if query:
    with st.spinner('Generating Response...'):
        response = rag_chain.invoke(query)
        st.write("Response:", response)
        
        compressed_docs = compression_retriever.get_relevant_documents(query)
        contexts = [doc.page_content for doc in compressed_docs]


        # --- Preparing Data for Evaluation ---
        data = {
            "query": [query],
            "response": [response],
            "context": [contexts],
        }

        dataset = Dataset.from_dict(data)
        df = pd.DataFrame(dataset)
        
         # Convert to dictionary
        df_dict = df.to_dict(orient='records')

        # Convert context to list
        for record in df_dict:
            if not isinstance(record.get('context'), list):
                if record.get('context') is None:
                    record['context'] = []
                else:
                    record['context'] = [record['context']]


        st.subheader("Evaluation")
        if st.button("Run Evaluation"):
            with st.spinner('Evaluating with Athina...'):
                # --- Evaluation in Athina AI ---
                from athina.keys import AthinaApiKey, OpenAiApiKey
                OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
                AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

                from athina.loaders import Loader
                dataset = Loader().load_dict(df_dict)

                from athina.evals import RagasContextRelevancy
                eval_results = RagasContextRelevancy(model="gpt-4o").run_batch(data=dataset).to_df()
                st.dataframe(eval_results)