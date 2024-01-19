import requests
import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# Chroma database path
CHROMA_PATH = "chroma2"

# Template for creating the prompt
PROMPT_TEMPLATE = """
Question: {question}
context:

{context}

"""

os.environ["OPENAI_API_KEY"] = "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"


def main():
    create_prompt()

def create_prompt(query_text):
    # Prepare the DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    try:
        results = db.similarity_search_with_relevance_scores(query_text, k=4)
        if len(results) == 0 or results[0][1] < 0.7:
            return "No matching results found."
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            return prompt_template.format(context=context_text, question=query_text)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    query_text = "People want to engage with the brands they love in new and more personal ways. This attitude is:"
    prompt = create_prompt(query_text)
    if prompt:
        print(prompt)

if __name__ == "__main__":
    main()
