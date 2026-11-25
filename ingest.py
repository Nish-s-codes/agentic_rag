from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from llm import llm


COLLECTION = "docs"

PREFIX = "Represent this sentence for searching relevant passages: "

MAX_RETRIES = 2


embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = QdrantClient(host="localhost", port=6333)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)  # small and fast on CPU


class State(TypedDict):

    question: str

    query: str  # current search query (may be rewritten)

    docs: List[dict]

    retries: int

    relevant: bool

    answer: str


def ask(prompt: str) -> str:

    return llm.invoke(prompt).content.strip()


def format_docs(docs):

    return "\n\n".join(

        f"SOURCE: {d['source']}, p.{d['page']}\n{d['text']}" for d in docs

    )


# ---- ROUTER (conditional edge at the start) ----

def route(state: State) -> str:

    answer = ask(

        "You have a library of PDF documents (research papers). "

        "Does answering the question below require searching that library? "

        "Reply with only DOCS or DIRECT. Use DIRECT only for greetings, "

        "simple math or general chit-chat.\n\n"

        f"Question: {state['question']}"

    )

    decision = "direct" if answer.upper().startswith("DIRECT") else "docs"

    print(f"[route] -> {decision}")

    return decision


# ---- NODES ----

def retrieve(state: State):

    print(f"[retrieve] query: {state['query']}")

    vec = embedder.encode(

        PREFIX + state["query"],

        normalize_embeddings=True,

    ).tolist()

    hits = client.query_points(

        COLLECTION,

        query=vec,

        limit=20,

    ).points

    docs = [h.payload for h in hits]

    scores = reranker.predict(

        [(state["query"], d["text"]) for d in docs]

    )

    ranked = sorted(

        zip(scores, docs),

        key=lambda x: x[0],

        reverse=True,

    )[:5]

    for s, d in ranked:

        print(

            f"   {d['source']} p.{d['page']} rerank={s:.2f}"

        )

    return {"docs": [d for _, d in ranked]}


def grade(state: State):

    answer = ask(

        f"Question: {state['question']}\n\nExcerpts:\n{format_docs(state['docs'])}\n\n"

        "Do these excerpts contain enough information to answer the question? "

        "Reply with only YES or NO."

    )

    relevant = answer.upper().startswith("YES")

    print(f"[grade] relevant: {relevant}")

    return {"relevant": relevant}


def rewrite(state: State):

    new_query = ask(

        "The search below did not find good results. Rewrite it as a better "

        "search query for a library of research papers. Use different keywords. "

        "Reply with only the new query.\n\n"

        f"Original question: {state['question']}\nPrevious query: {state['query']}"

    )

    print(f"[rewrite] new query: {new_query}")

    return {"query": new_query, "retries": state["retries"] + 1}


def generate(state: State):

    print("[generate]")

    answer = ask(

        "Answer the question using ONLY the excerpts below. Be specific and include "

        "concrete mechanisms, numbers and results when the excerpts contain them. "

        "Cite every claim with its SOURCE like (file.pdf, p.3), using the file name, "

        "never an excerpt number. If the excerpts do not contain the answer, say so "

        "clearly instead of guessing.\n\n"

        f"Excerpts:\n{format_docs(state['docs'])}\n\nQuestion: {state['question']}"

    )

    return {"answer": answer}


def direct(state: State):

    print("[direct]")

    return {"answer": ask(state["question"])}


# ---- CONDITIONAL EDGE after grading ----

def after_grade(state: State) -> str:

    if state["relevant"] or state["retries"] >= MAX_RETRIES:

        return "generate"

    return "rewrite"


# ---- BUILD THE GRAPH ----

g = StateGraph(State)

g.add_node("retrieve", retrieve)

g.add_node("grade", grade)

g.add_node("rewrite", rewrite)

g.add_node("generate", generate)

g.add_node("direct", direct)


g.add_conditional_edges(START, route, {"docs": "retrieve", "direct": "direct"})

g.add_edge("retrieve", "grade")

g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})

g.add_edge("rewrite", "retrieve")

g.add_edge("generate", END)

g.add_edge("direct", END)


app = g.compile()


if __name__ == "__main__":

    while True:

        q = input("\nQuestion (or 'quit'): ").strip()

        if q.lower() in ("quit", "exit", ""):

            break

        result = app.invoke(

            {"question": q, "query": q, "docs": [], "retries": 0, "relevant": False, "answer": ""}

        )

        print("\nANSWER:\n", result["answer"])

        print("\nSOURCES:")

        for d in result["docs"]:

            print(f" - {d['source']} p.{d['page']}")
