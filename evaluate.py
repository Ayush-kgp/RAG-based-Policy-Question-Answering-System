from rag_pipeline import answer_question


def get_evaluation_questions():
    """
    Returns a list of evaluation questions with expected answer types.
    """
    return [
        {
            "question": "Can I change the declared value after placing my order?",
            "expected": "answerable"
        },
        {
            "question": "What happens if I raise a claim after 3 days of delivery?",
            "expected": "answerable"
        },
        {
            "question": "Does Delhivery compensate for shipment delays?",
            "expected": "partially_answerable"
        },
        {
            "question": "What personal information does Delhivery collect?",
            "expected": "answerable"
        },
        {
            "question": "Who is the Data Protection Officer at Delhivery?",
            "expected": "unanswerable"
        },
        {
            "question": "Which courier partner handles intracity deliveries?",
            "expected": "unanswerable"
        }
    ]


def evaluate_system(vectorstore):
    """
    Runs evaluation questions against the RAG system and prints results.
    """

    questions = get_evaluation_questions()

    print("\nRAG Evaluation\n")

    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        expected = item["expected"]

        answer = answer_question(vectorstore, question)

        print(f"Q{idx}: {question}")
        print(f"Expected: {expected}")
        print(f"Model Answer: {answer}")
        print("-" * 80)


if __name__ == "__main__":
    from data_loader import load_all_documents
    from vector_store import create_vector_store

    chunks = load_all_documents()
    vectorstore = create_vector_store(chunks)

    evaluate_system(vectorstore)
