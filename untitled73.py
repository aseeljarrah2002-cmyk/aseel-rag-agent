import os

from flask import Flask, request, jsonify, render_template
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI


# =========================================================
# Flask
# =========================================================

app = Flask(__name__)


# =========================================================
# Azure AI Search
# =========================================================

SEARCH_ENDPOINT = "https://aisearchtraining0.search.windows.net"

SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]

SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]


# =========================================================
# Azure OpenAI
# =========================================================

OPENAI_ENDPOINT = "https://openai-training.openai.azure.com/"

OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]

OPENAI_DEPLOYMENT = "gpt-5.4-mini"

EMBEDDING_DEPLOYMENT = os.environ[
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
]


# =========================================================
# Azure Search Vector Field
# =========================================================

VECTOR_FIELD = os.environ.get(
    "AZURE_SEARCH_VECTOR_FIELD",
    "contentVector"
)


# =========================================================
# Azure AI Search Client
# =========================================================

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)


# =========================================================
# Azure OpenAI Client
# =========================================================

openai_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2025-04-01-preview"
)


# =========================================================
# Home Page
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# =========================================================
# Health Check
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "message": "RAG API is running"
    })


# =========================================================
# Ask
# =========================================================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data or "question" not in data:

        return jsonify({
            "error": "Please provide a question."
        }), 400


    question = data["question"].strip()

    if not question:

        return jsonify({
            "error": "Question cannot be empty."
        }), 400


    try:

        # =================================================
        # 1. Create embedding for the question
        # =================================================

        embedding_response = openai_client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=question
        )

        query_vector = embedding_response.data[0].embedding


        # =================================================
        # 2. Create Vector Query
        # =================================================

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=5,
            fields=VECTOR_FIELD
        )


        # =================================================
        # 3. Hybrid Search
        # Keyword + Vector
        # =================================================

        results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            top=5
        )


        # =================================================
        # 4. Build Context
        # =================================================

        context_parts = []

        sources = []

        for result in results:

            content = result.get("content")

            if content:

                source = result.get(
                    "source",
                    result.get(
                        "metadata_storage_name",
                        "Unknown"
                    )
                )

                page = result.get(
                    "page",
                    result.get(
                        "metadata_storage_path",
                        "Unknown"
                    )
                )

                context_parts.append(
                    f"""
Source: {source}
Page: {page}

Content:
{content}
"""
                )

                sources.append({
                    "source": source,
                    "page": page
                })


        # =================================================
        # 5. Check if Search found anything
        # =================================================

        if not context_parts:

            return jsonify({
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the provided documents."
                ),
                "sources": []
            })


        context = "\n\n".join(context_parts)


        # =================================================
        # 6. Send Context to Azure OpenAI
        # =================================================

        response = openai_client.chat.completions.create(

            model=OPENAI_DEPLOYMENT,

            messages=[

                {
                    "role": "system",

                    "content": """
You are a RAG assistant.

Answer the user's question using ONLY
the information provided in the context.

The context comes from five provided
Word documents.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not contained in
   the context, say:

   "The answer is not available
   in the provided documents."

4. Give a clear and concise answer.
5. Base the answer only on retrieved
   document content.
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


        # =================================================
        # 7. Get Answer
        # =================================================

        answer = response.choices[0].message.content


        # =================================================
        # 8. Return Answer + Sources
        # =================================================

        return jsonify({

            "question": question,

            "answer": answer,

            "sources": sources

        })


    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        )
    )



    


       
    


    


    

    
    

       
            
        


    

    
