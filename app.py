import os

from flask import Flask, request, jsonify, render_template
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI

app = Flask(_name_)


# =========================
# Environment Variables
# =========================

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]


# =========================
# Azure AI Search
# =========================

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)


# =========================
# Azure OpenAI
# =========================

openai_client = OpenAI(
    api_key=OPENAI_KEY,
    base_url=OPENAI_ENDPOINT
)



# =========================
# Test endpoint
# =========================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")



# =========================
# RAG endpoint
# =========================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({
            "error": "Please provide a question"
        }), 400

    question = data["question"]

    # Search Azure AI Search
    results = search_client.search(
        search_text=question,
        top=5
    )

    context_parts = []

    for result in results:
        content = result.get("snippet")

        if content:
            context_parts.append(content)

    context = "\n\n".join(context_parts)

    if not context:
        return jsonify({
            "answer": "I could not find relevant information in the documents."
        })

    # Send context to Azure OpenAI
    response = openai_client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": """
You are a RAG assistant.

Answer the user's question using ONLY the provided context.

If the answer is not contained in the context,
say that you don't know.

Do not make up information.
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    return jsonify({
        "question": question,
        "answer": answer
    })


if _name_ == "_main_":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
