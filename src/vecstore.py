from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

text = TextLoader(
    "./state_of_the_union_2023.txt"
).load()

docs = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
).split_documents(text)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(), persist_directory="./chroma_db"
)

vectorstore.persist()