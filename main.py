from rag import ask_rag

def main():
    print("=" * 50)
    print("Basic RAG System")
    print("=" * 50)

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            answer = ask_rag(question)

            print(f"\nAssistant: {answer}")

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
