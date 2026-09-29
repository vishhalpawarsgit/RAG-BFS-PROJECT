import logging

from rag_chain import RAGChain


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------------------------------
# Main Application
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("   BFS DOMAIN-SPECIFIC DOCUMENT QA SYSTEM")
    print("   Retrieval-Augmented Generation (RAG)")
    print("=" * 60)

    print("\nWelcome! 👋")
    print("Ask questions about the BFS documents.")
    print("Type 'exit' or 'quit' to close the application.")
    print("-" * 60)

    # --------------------------------------------------
    # Initialize RAG system
    # --------------------------------------------------

    try:

        rag = RAGChain()

    except Exception as e:

        print("\n❌ Failed to initialize the RAG system.")
        print(f"Error: {e}")

        logging.exception("RAG initialization failed.")

        return

    print("\n✅ RAG system ready!")
    print("-" * 60)

    # --------------------------------------------------
    # Continuous Question-Answer Loop
    # --------------------------------------------------

    while True:

        try:

            query = input("\nYou: ").strip()

        except KeyboardInterrupt:

            print("\n\n👋 Application stopped by user.")
            break

        except EOFError:

            print("\n\n👋 Application closed.")
            break

        # --------------------------------------------------
        # Empty input handling
        # --------------------------------------------------

        if not query:

            print("⚠️ Please enter a question.")

            continue

        # --------------------------------------------------
        # Exit handling
        # --------------------------------------------------

        if query.lower() in {"exit", "quit"}:

            print("\n👋 Thank you for using the BFS RAG system.")
            break

        # --------------------------------------------------
        # Process question
        # --------------------------------------------------

        print("\n🔎 Searching BFS documents...")

        try:

            result = rag.ask(query)

        except Exception as e:

            logging.exception("Error processing query.")

            print(
                "\n❌ Sorry, an error occurred while "
                "processing your question."
            )

            print(f"Error: {e}")

            continue

        # --------------------------------------------------
        # Display answer
        # --------------------------------------------------

        print("\n🤖 Answer:")
        print(result["answer"])

        # --------------------------------------------------
        # Display sources
        # --------------------------------------------------

        sources = result.get("sources", [])

        if sources:

            print("\n📚 Sources:")

            for source in sources:

                print(f"   • {source}")

        print("\n" + "-" * 60)


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()