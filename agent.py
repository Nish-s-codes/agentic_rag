import re
import json
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

from llm import llm


COLLECTION = "docs"
SUMMARIES = "doc_summaries"
PREFIX = "Represent this sentence for searching relevant passages: "
MAX_RETRIES = 2
NOT_IN_LIBRARY = -8.0
CHAT_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))[\s!.]*$"
    r"|^[\d\s+\-*/().=?x]+$",
    re.I,
)


embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(host="localhost", port=6333)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class State(TypedDict):
    question: str
    subs: List[str]           # sub-questions, fixed after decomposition
    missing: List[str]        # sub-questions not covered
    retryable: List[str]      # missing ones worth retrying
    retry_queries: List[str]  # rewritten queries for the retryable ones
    sub_best: dict            # best rerank score per sub-question
    docs: List[dict]
    retries: int
    answer: str
    top_score: float


def ask(prompt: str) -> str:
    return llm.invoke(prompt).content.strip()


def format_docs(docs):
    return "\n\n".join(
        f"SOURCE: {d['source']}, p.{d['page']}\n{d['text']}"
        for d in docs
    )


# ---- DOCUMENT-LEVEL INDEX HELPERS ----

def doc_candidates(question: str, k: int = 5) -> List[dict]:
    if not client.collection_exists(SUMMARIES):
        return []
    vec = embedder.encode(PREFIX + question, normalize_embeddings=True).tolist()
    hits = client.query_points(SUMMARIES, query=vec, limit=k).points
    return [h.payload for h in hits]


def find_named_doc(question: str):
    if not client.collection_exists(SUMMARIES):
        return None
    pts, _ = client.scroll(SUMMARIES, limit=2000, with_payload=True)
    for p in pts:
        if p.payload["source"].replace(".pdf", "") in question:
            return p.payload
    return None


# ---- ROUTER ----

def route(state: State) -> str:
    q = state["question"]

    if CHAT_RE.match(q.strip()):
        print("[route] -> direct")
        return "direct"

    cands = doc_candidates(q)

    if not cands:
        print("[route] -> docs (no summaries found, run summarize.py)")
        return "docs"

    listing = "\n\n".join(
        f"{c['source']}: {c['summary']}" for c in cands
    )

    kind = ask(
        "You route messages for a question-answering system over a library of research PDFs.\n"
        "Reply with exactly one word:\n"
        "CHAT - greeting, thanks, or simple arithmetic\n"
        "SUMMARY - asks for a summary or overview of a specific document, or what it is about\n"
        "DOCS - could be answered using the documents below\n"
        "GENERAL - none of the documents below are relevant\n\n"
        f"Closest documents in the library:\n{listing}\n\n"
        f"Message: {q}"
    ).upper()

    if kind.startswith("CHAT"):
        decision = "direct"
    elif kind.startswith("SUMMARY"):
        decision = "summary"
    elif kind.startswith("GENERAL"):
        decision = "general"
    else:
        decision = "docs"   # when unsure, search the library

    print(f"[route] -> {decision}")

    return decision


# ---- QUERY DECOMPOSITION ----

def decompose(question: str) -> List[str]:
    out = ask(
        "Split the question into independent sub-questions, each about ONE topic "
        "so it can be searched separately. If it is already a single question, "
        "return it unchanged. Return ONLY a JSON list of strings, max 3 items.\n\n"
        f"Question: {question}"
    )

    try:
        qs = json.loads(
            out[out.index("["): out.rindex("]") + 1]
        )

        qs = [
            q for q in qs
            if isinstance(q, str) and q.strip()
        ][:3]

        return qs or [question]

    except Exception:
        return [question]


# ---- NODES ----

def retrieve(state: State):
    if state["retries"] == 0:
        queries = decompose(state["question"])
        subs = queries
    else:
        queries = state["retry_queries"]
        subs = state["subs"]

    print(f"[retrieve] queries: {queries}")

    per_query = {1: 5, 2: 3, 3: 2}[len(queries)]

    merged = list(state["docs"])  # keep chunks found earlier
    seen = {
        (d["source"], d["page"], d["text"][:50])
        for d in merged
    }
    sub_best = dict(state["sub_best"])
    best = -99.0

    for i, q in enumerate(queries):
        vec = embedder.encode(
            PREFIX + q,
            normalize_embeddings=True,
        ).tolist()

        hits = client.query_points(
            COLLECTION,
            query=vec,
            limit=20,
        ).points

        docs = [h.payload for h in hits]

        if not docs:
            continue

        scores = reranker.predict(
            [(q, d["text"]) for d in docs]
        )

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True,
        )[:per_query]

        top = float(ranked[0][0])
        best = max(best, top)

        owner = subs[i] if state["retries"] == 0 else state["retryable"][i]
        sub_best[owner] = max(sub_best.get(owner, -99.0), top)

        for s, d in ranked:
            key = (
                d["source"],
                d["page"],
                d["text"][:50],
            )

            if key in seen:
                continue

            seen.add(key)
            merged.append({**d, "score": float(s)})

            print(
                f"   {d['source']} p.{d['page']} "
                f"rerank={s:.2f}"
            )

    return {
        "docs": merged,
        "top_score": best,
        "subs": subs,
        "sub_best": sub_best,
    }


def grade(state: State):
    subs = state["subs"]

    numbered = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(subs)
    )

    out = ask(
        f"Sub-questions:\n{numbered}\n\n"
        f"Excerpts:\n{format_docs(state['docs'])}\n\n"
        "For each sub-question, decide whether the excerpts directly contain the "
        "information needed to answer it. Return ONLY a JSON list of true/false "
        f"values, exactly {len(subs)} items, in order."
    )

    try:
        flags = json.loads(
            out[out.index("["): out.rindex("]") + 1]
        )

        flags = [
            f is True or str(f).lower() == "true"
            for f in flags
        ]

        if len(flags) != len(subs):
            raise ValueError

    except Exception:
        flags = [True] * len(subs)  # unparseable: don't loop, let generate be honest

    missing = [s for s, ok in zip(subs, flags) if not ok]

    # only retry sub-questions that scored above the "clearly not in library" level
    retryable = [
        s for s in missing
        if state["sub_best"].get(s, 0.0) >= NOT_IN_LIBRARY
    ]

    print(f"[grade] covered: {flags}")

    if missing and not retryable:
        print("[grade] missing parts look absent from the library, skipping retry")

    return {"missing": missing, "retryable": retryable}


def rewrite(state: State):
    new = []

    for m in state["retryable"]:
        new.append(
            ask(
                "The search below did not find good results. Rewrite it as ONE short "
                "natural-language search query (under 20 words). No AND/OR operators, no "
                "quotation marks. Keep technical names exactly as written and do not "
                f"expand acronyms.\n\nQuestion: {m}"
            )
        )

    print(f"[rewrite] {new}")

    return {
        "retry_queries": new,
        "retries": state["retries"] + 1,
    }


def generate(state: State):
    print("[generate]")

    answer = ask(
        "Answer the question using ONLY the excerpts below. If the question has several "
        "parts, answer each part separately. Be specific and include concrete mechanisms, "
        "numbers and results when the excerpts contain them. Cite every claim with its "
        "SOURCE like (file.pdf, p.3), using the file name, never an excerpt number. For any "
        "part the excerpts do not cover, say clearly that it is not in the documents "
        "instead of guessing.\n\n"
        f"Excerpts:\n{format_docs(state['docs'])}\n\n"
        f"Question: {state['question']}"
    )

    return {"answer": answer}


def direct(state: State):
    print("[direct]")

    return {
        "answer": ask(state["question"])
    }


def summary(state: State):
    print("[summary]")

    doc = find_named_doc(state["question"])

    if doc:
        return {"answer": f"{doc['source']}\n{doc['summary']}"}

    cands = doc_candidates(state["question"], k=3)

    if not cands:
        return {"answer": "No document summaries found. Run summarize.py first."}

    best, others = cands[0], cands[1:]

    text = f"Closest match: {best['source']}\n{best['summary']}"

    if others:
        text += "\n\nOther possible matches: " + ", ".join(
            c["source"] for c in others
        )

    return {"answer": text}


def general(state: State):
    print("[general] library does not cover this")

    note = (
        "(None of your documents cover this topic. This answer comes from the "
        "model's general knowledge, not your library, and may be outdated.)\n\n"
    )

    return {"answer": note + ask(state["question"])}


# ---- CONDITIONAL EDGE AFTER GRADING ----

def after_grade(state: State) -> str:
    if not state["retryable"] or state["retries"] >= MAX_RETRIES:
        return "generate"

    return "rewrite"


# ---- BUILD THE GRAPH ----

g = StateGraph(State)

g.add_node("retrieve", retrieve)
g.add_node("grade", grade)
g.add_node("rewrite", rewrite)
g.add_node("generate", generate)
g.add_node("direct", direct)
g.add_node("summary", summary)
g.add_node("general", general)

g.add_conditional_edges(
    START,
    route,
    {
        "docs": "retrieve",
        "direct": "direct",
        "summary": "summary",
        "general": "general",
    },
)

g.add_edge("retrieve", "grade")

g.add_conditional_edges(
    "grade",
    after_grade,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

g.add_edge("rewrite", "retrieve")
g.add_edge("generate", END)
g.add_edge("direct", END)
g.add_edge("summary", END)
g.add_edge("general", END)

app = g.compile()


# ---- CLI ----

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (or 'quit'): ").strip()

        if q.lower() in ("quit", "exit", ""):
            break

        result = app.invoke(
            {
                "question": q,
                "subs": [],
                "missing": [],
                "retryable": [],
                "retry_queries": [],
                "sub_best": {},
                "docs": [],
                "retries": 0,
                "answer": "",
                "top_score": 0.0,
            }
        )

        print("\nANSWER:\n", result["answer"])

        good = sorted({
            (d["source"], d["page"])
            for d in result["docs"]
            if d.get("score", 0) > 0
        })

        if good:
            print("\nSOURCES:")

            for source, page in good:
                print(f" - {source} p.{page}")