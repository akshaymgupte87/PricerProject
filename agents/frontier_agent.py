import os
from pathlib import Path
from typing import List, Dict
from litellm import completion
from sentence_transformers import SentenceTransformer
from agents.agent import Agent
from agents.guardrails import extract_single_price
from agents.retrieval import (
    CrossEncoderReranker,
    RetrievalCandidate,
    RetrievalResult,
    SQLiteBM25Index,
    reciprocal_rank_fusion,
    rerank,
    expand_product_query,
)


def local_ollama_model(value: str) -> str:
    """Normalize an Ollama tag into the provider/model form LiteLLM requires."""
    value = value.strip()
    return value if value.startswith("ollama/") else f"ollama/{value}"


DEFAULT_MODEL = local_ollama_model(os.getenv("PRICER_FRONTIER_MODEL", "qwen3.6:latest"))
DEFAULT_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
DEFAULT_RERANKER_MODEL = os.getenv(
    "PRICER_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
).strip()
DEFAULT_BM25_PATH = Path(__file__).resolve().parents[1] / "products_bm25.sqlite3"


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = DEFAULT_MODEL

    def __init__(
        self,
        collection,
        model_name=DEFAULT_MODEL,
        api_base=DEFAULT_API_BASE,
        *,
        encoder=None,
        lexical_index=None,
        candidate_reranker=None,
        candidate_count: int = 20,
        result_count: int = 5,
    ):
        """
        Set up this instance with local Qwen through Ollama, the Chroma datastore,
        and the vector encoding model. No cloud API key is required.
        """
        self.log("Initializing Frontier Agent")
        self.MODEL = local_ollama_model(model_name)
        self.api_base = api_base
        self.log(f"Frontier Agent is using local model {self.MODEL}")
        self.collection = collection
        self.model = encoder or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.lexical_index = lexical_index or SQLiteBM25Index(
            os.getenv("PRICER_BM25_PATH", str(DEFAULT_BM25_PATH))
        )
        if candidate_reranker is not None:
            self.reranker = candidate_reranker
        elif DEFAULT_RERANKER_MODEL.lower() in {"", "none", "off"}:
            self.reranker = None
        else:
            self.reranker = CrossEncoderReranker(DEFAULT_RERANKER_MODEL)
        self.candidate_count = max(candidate_count, result_count)
        self.result_count = result_count
        self.last_retrieval: RetrievalResult | None = None
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(
        self, description: str, similars: List[str], prices: List[float]
    ) -> List[Dict[str, str]]:
        """
        Create the message list for the local Qwen call.
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]

    def retrieve(
        self,
        description: str,
        *,
        metadata_filter: dict | None = None,
        query_variants: List[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve, fuse, and rerank comparable products."""
        variants = query_variants or expand_product_query(description)
        rankings = []
        for variant in variants:
            vector = self.model.encode([variant])
            query_args = {
                "query_embeddings": vector.astype(float).tolist(),
                "n_results": self.candidate_count,
            }
            if metadata_filter:
                query_args["where"] = metadata_filter
            dense_results = self.collection.query(**query_args)
            ids = dense_results.get("ids", [[]])[0]
            documents = dense_results.get("documents", [[]])[0]
            metadatas = dense_results.get("metadatas", [[]])[0]
            rankings.append([
                RetrievalCandidate(
                    id=str(item_id),
                    document=document or "",
                    metadata=metadata or {},
                    dense_rank=rank,
                )
                for rank, (item_id, document, metadata) in enumerate(
                    zip(ids, documents, metadatas), start=1
                )
            ])
            lexical = self.lexical_index.search(
                variant,
                limit=self.candidate_count,
                metadata_filter=metadata_filter,
            )
            if lexical:
                rankings.append(lexical)
        fused = reciprocal_rank_fusion(rankings)
        reranked = rerank(description, fused, self.reranker)
        result = RetrievalResult(
            candidates=reranked[: self.result_count],
            used_hybrid=any(item.lexical_rank is not None for item in fused),
            reranked=self.reranker is not None,
        )
        self.last_retrieval = result
        return result

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        result = self.retrieve(description)
        documents = [item.document for item in result.candidates]
        prices = [float(item.metadata["price"]) for item in result.candidates]
        mode = "hybrid" if result.used_hybrid else "dense fallback"
        reranking = " with reranking" if result.reranked else ""
        self.log(f"Frontier Agent found similar products using {mode}{reranking}")
        return documents, prices

    def evidence(self) -> list[dict]:
        """Return serializable citations for the most recent pricing request."""
        if not self.last_retrieval:
            return []
        return [
            {
                "id": item.id,
                "description": item.document,
                "price": float(item.metadata.get("price", 0)),
                "category": item.metadata.get("category"),
                "score": item.reranker_score
                if item.reranker_score is not None
                else item.fusion_score,
            }
            for item in self.last_retrieval.candidates
        ]

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        return extract_single_price(s)

    def price(self, description: str) -> float:
        """
        Call local Qwen through Ollama to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        documents, prices = self.find_similars(description)
        self.log(
            f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products"
        )
        response = completion(
            model=self.MODEL,
            api_base=self.api_base,
            messages=self.messages_for(description, documents, prices),
            temperature=0,
            max_tokens=50,
            num_ctx=4096,
            think=False,
            keep_alive="30m",
            seed=42,
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
