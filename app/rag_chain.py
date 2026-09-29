import os
import logging
import re

from dotenv import load_dotenv
from openai import OpenAI

from retriever import Retriever


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "3"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class RAGChain:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please add it to your .env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.retriever = Retriever(top_k=TOP_K)

        logging.info("RAG chain initialized.")

    def _build_context(self, results):
        context_parts = []

        for i, result in enumerate(results, start=1):
            source = result.get("source", "Unknown")
            page = result.get("page", "Unknown")
            text = result.get("text", "").strip()

            context_parts.append(
                f"""
SOURCE {i}
Document: {source}
Page: {page}

Content:
{text}
"""
            )

        return "\n".join(context_parts)

    def _extract_period_from_query(self, query):
        patterns = [
            r"\b(for|in)\s+(H[12]\s+FY\d{4})\b",
            r"\b(for|in)\s+(FY\d{4})\b",
            r"\b(at|on|as of)\s+"
            r"(\d{1,2}\s+"
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{4})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return f"{match.group(1).lower()} {match.group(2)}"

        return None

    def _extract_explicit_fact(self, query, results):
        query_lower = query.lower()

        comparison_terms = [
            "compared with",
            "compared to",
            "change",
            "changed",
            "increase",
            "increased",
            "decrease",
            "decreased",
            "growth",
            "grew",
            "decline",
        ]

        if any(term in query_lower for term in comparison_terms):
            return None

        financial_terms = [
            "net profit",
            "net interest income",
            "interest income",
            "interest expense",
            "non-interest income",
            "operating expenses",
            "profit before tax",
            "total assets",
            "customer deposits",
            "shareholders' equity",
            "loan-to-deposit ratio",
            "cost-to-income ratio",
            "return on equity",
        ]

        matched_term = next(
            (term for term in financial_terms if term in query_lower),
            None
        )

        if not matched_term:
            return None

        for result in results:
            text = result.get("text", "")

            # Handles common PDF extraction variants such as:
            # Net profit: ₹1,360 crore
            # Net profit: ₹ 1,360 crore
            # Net Profit (H1 FY2026): ₹1,360 crore
            # Loan-to-deposit ratio: 79.7%
            pattern = re.compile(
                rf"{re.escape(matched_term)}"
                rf"(?:\s*\([^)]*\))?"
                rf"\s*:\s*"
                rf"((?:₹\s*)?"
                rf"[0-9][0-9,]*(?:\.[0-9]+)?"
                rf"(?:\s*%)?"
                rf"(?:\s+(?:crore|million|billion))?"
                rf"(?:\s+currency\s+units)?"
                rf")",
                re.IGNORECASE,
            )

            match = pattern.search(text)

            if match:
                value = re.sub(r"₹\s+", "₹", match.group(1).strip())

                return {
                    "term": matched_term,
                    "value": value,
                    "source": result.get("source", "Unknown document"),
                    "page": result.get("page", "Unknown"),
                }

        return None

    def ask(self, query):
        if not query or not query.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
            }

        query = query.strip()

        logging.info("Processing query: %s", query)

        results = self.retriever.retrieve(query, top_k=TOP_K)

        if not results:
            return {
                "answer": (
                    "The information is not available "
                    "in the provided documents."
                ),
                "sources": [],
            }

        logging.info("Retrieved %d relevant chunks.", len(results))

        explicit_fact = self._extract_explicit_fact(query, results)

        if explicit_fact:
            period = self._extract_period_from_query(query)

            if period:
                answer = (
                    f"The {explicit_fact['term']} "
                    f"{period} was {explicit_fact['value']}."
                )
            else:
                answer = (
                    f"The {explicit_fact['term']} "
                    f"was {explicit_fact['value']}."
                )

            source = (
                f"{explicit_fact['source']}, "
                f"page {explicit_fact['page']}"
            )

            logging.info(
                "Explicit financial fact found: %s = %s",
                explicit_fact["term"],
                explicit_fact["value"],
            )

            return {
                "answer": answer,
                "sources": [source],
            }

        context = self._build_context(results)

        system_prompt = """
You are a Banking & Financial Services (BFS)
document question-answering assistant.

Answer the user's question using ONLY the supplied
document context.

Rules:

1. Read all retrieved sources carefully.

2. If the answer is explicitly stated in the context,
   use that information directly.

3. Never claim that information is unavailable when
   the answer is explicitly present in the context.

4. For financial values, preserve the exact number,
   percentage, currency unit, and financial year
   stated in the document.

5. For comparison questions, answer the comparison
   requested by the user. Do not substitute an absolute
   value for a requested change or percentage.

6. Do not invent financial facts.

7. Do not use outside knowledge.

8. If the requested information genuinely does not
   appear in the supplied context, say:

   "The information is not available in the provided documents."

9. Keep answers concise.

10. Mention the relevant document and page when useful.

11. If the supplied documents are synthetic samples,
    do not present their figures or policies as real-world
    financial information or financial advice.
"""

        user_prompt = f"""
DOCUMENT CONTEXT
================

{context}

================

USER QUESTION
================

{query}

================

Answer using only the document context.
"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            answer = response.choices[0].message.content.strip()

        except Exception as e:
            logging.exception("OpenAI API error")

            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
            }

        sources = []

        for result in results:
            source_info = (
                f"{result.get('source')}, page {result.get('page')}"
            )

            if source_info not in sources:
                sources.append(source_info)

        return {
            "answer": answer,
            "sources": sources,
        }


def main():
    print("\n========================================")
    print(" BFS RAG Question Answering System")
    print("========================================")
    print("Type 'exit' or 'quit' to stop.\n")

    try:
        rag = RAGChain()
    except Exception as e:
        print(f"\n❌ Error initializing RAG system: {e}")
        return

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        result = rag.ask(query)

        print("\nAssistant:")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")

        print()


if __name__ == "__main__":
    main()
