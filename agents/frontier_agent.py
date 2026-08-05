import re
import os
from typing import List, Dict
from litellm import completion
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


def local_ollama_model(value: str) -> str:
    """Normalize an Ollama tag into the provider/model form LiteLLM requires."""
    value = value.strip()
    return value if value.startswith("ollama/") else f"ollama/{value}"


DEFAULT_MODEL = local_ollama_model(os.getenv("PRICER_FRONTIER_MODEL", "qwen3.6:latest"))
DEFAULT_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = DEFAULT_MODEL

    def __init__(self, collection, model_name=DEFAULT_MODEL, api_base=DEFAULT_API_BASE):
        """
        Set up this instance with local Qwen through Ollama, the Chroma datastore,
        and the vector encoding model. No cloud API key is required.
        """
        self.log("Initializing Frontier Agent")
        self.MODEL = local_ollama_model(model_name)
        self.api_base = api_base
        self.log(f"Frontier Agent is using local model {self.MODEL}")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
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

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

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
