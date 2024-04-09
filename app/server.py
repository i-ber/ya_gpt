from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

import os

from yandex_chain import YandexEmbeddings, ChatYandexGPT, YandexLLM
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
import chromadb

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.pydantic_v1 import BaseModel

app = FastAPI(
    title="YaGPT LangServe Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


ya_auth = {
    "api_key": os.getenv("YA_API_KEY"),
    "folder_id": os.getenv("YA_FOLDER_ID")
}

collection_name = "botpress_collection"

namespace = "test1"
temp = 0.1
tokens = 100
system_prompt = """Ты ассистент отеля, в твоей базе есть следующая информация по вопросу клиента,
ты можешь её использовать для ответа:
{context}
"""


embeddings = YandexEmbeddings(**ya_auth)
chroma_client = chromadb.HttpClient(host=os.getenv("CHROMADB_HOST"), port=int(os.getenv("CHROMADB_PORT")))
langchain_chroma = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embeddings,
)
retriever = langchain_chroma.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """Ты ассистент отеля, в твоей базе есть следующая информация по вопросу клиента,
ты можешь её использовать для ответа:
{context}

Вопрос: {input}

Ответ:"""
prompt = PromptTemplate.from_template(template)
llm = YandexLLM(**ya_auth, temperature=temp, max_tokens=tokens)
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
