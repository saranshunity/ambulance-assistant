import re
import glob
import os

from fastembed import TextEmbedding
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter


class LocalEmbeddings(Embeddings):
    """Wraps the same local bge-small model you used in the sample project."""

    def __init__(self):
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text):
        return list(self.model.query_embed([text]))[0].tolist()


def load_faq(folder="faq"):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")]
    )
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for d in splitter.split_text(text):
            title = d.metadata.get("title", "")
            section = d.metadata.get("section", "")
            d.page_content = f"{title} > {section}\n\n{d.page_content}"
            d.metadata["source"] = os.path.basename(path)
            docs.append(d)
    return docs


DOCS = load_faq()
_store = InMemoryVectorStore.from_documents(DOCS, LocalEmbeddings())
_vector = _store.as_retriever(search_kwargs={"k": 4})
def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


_bm25 = BM25Retriever.from_documents(DOCS, k=4, preprocess_func=tokenize)


def search_faq_docs(query: str, k: int = 3):
    """Hybrid search: vector + BM25, merged with reciprocal rank fusion."""
    scores, by_key = {}, {}
    for retriever in (_vector, _bm25):
        for rank, d in enumerate(retriever.invoke(query)):
            key = (d.metadata["source"], d.page_content[:60])
            by_key[key] = d
            scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [by_key[key] for key in best]


def search_faq(question: str) -> dict:
    """Search the service FAQ (ambulance types, fares and payment, booking, cancellation, documents). Use for any question about how the service works. Not for medical questions."""
    docs = search_faq_docs(question)
    return {
        "passages": [
            {"source": d.metadata["source"], "text": d.page_content} for d in docs
        ]
    }