import os
import numpy as np
import faiss   #successful

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_openai import ChatOpenAI
from openai import OpenAI
from pypdf import PdfReader
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class HunyuanEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("HUNYUAN_API_KEY"),
            base_url="https://tokenhub.tencentmaas.com/v1",
        )
        self.model = "kinfra-text-embedding-0.6b"

    def embed_documents(self, texts):
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


embeddings = HunyuanEmbeddings()

def load_pdf(file_path):
    reader = PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(text, metadata={"source": file_path, "page": i+1}))
    return docs

def load_markdowns(folder):
    docs = []
    for path in glob.glob(f"{folder}/**/*.md", recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            docs.append(Document(f.read(), metadata={"source": path}))
    return docs

# -------- 切片 --------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    return splitter.split_documents(docs)




documents = []
#documents.extend(load_pdf("./test.pdf"))
documents.extend(load_markdowns("./docs"))
documents = split_docs(documents)


def build_faiss_index(docs, embed_model):
    vectors = np.array(
        [embed_model.embed_query(d.page_content) for d in docs],
        dtype="float32"
    )
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, docs


index, docs = build_faiss_index(documents, embeddings)

def retrieve(query: str, k: int = 2):
    q_vec = np.array([embeddings.embed_query(query)], dtype="float32")
    D, I = index.search(q_vec, k)
    return [docs[i] for i in I[0]]

llm = ChatOpenAI(
    model="hy3",
    openai_api_key=os.getenv("HUNYUAN_API_KEY"),
    openai_api_base="https://tokenhub.tencentmaas.com/v1",
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是严谨的 AI 工程师，只根据上下文回答，不知道就说不知道。"),
    ("user", "上下文：\n{context}\n\n问题：{question}"),
])

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)


rag_chain = (
    {
        "context": lambda x: format_docs(retrieve(x["question"])),
        "question": lambda x: x["question"],
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "RESTful服务有哪些HTTP方法"
    answer = rag_chain.invoke({"question": question})

    print("问题：", question)
    print("回答：", answer)
