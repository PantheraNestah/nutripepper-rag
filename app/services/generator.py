def generate_response(query: str, context: list):
    context_text = " ".join(context)

    response = f"Based on verified nutrition knowledge: {context_text} This information addresses your question about '{query}'."

    return response