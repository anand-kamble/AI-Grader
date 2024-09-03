# https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/
# %%
import os
from re import template
from tabnanny import verbose
from typing import List

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models.ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document

# %%
file_path = "./solutions_questions/Question Bank 1 (Tan et al 2nd Edition).pdf"
# %%
loader = PyPDFLoader(file_path)
pages: List[Document] = loader.load_and_split()

# %%

ollama_emb = OllamaEmbeddings(
    model="llama3.1",
)


embed: List[List[float]] = ollama_emb.embed_documents(
    list(map(lambda p: p.page_content, pages))
)
# %%
template: str = """You are a teacher who is creating a list of questions and answers. 
You are generating a usable json from 
the given text. The json has array of object with 
two keys: question, answer.
Given text is: {text}
json:
"""

prompt = PromptTemplate.from_template(template)

chat_llm = ChatOllama(
    model="llama3.1",
    # model="llama2:7b-chat",
    temperature=0.7,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)


chain = LLMChain(prompt=prompt, llm=chat_llm, verbose=True)

response = chain.predict(text=pages[0].page_content)
# %%
llm.chain
# %%
os.listdir("./")
# %%
