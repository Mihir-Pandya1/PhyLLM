import os
import pickle
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Paths ===
simcse_model_path = "/home/Group_1/FineTune_SimCSE/SciBERT_SimCSE/"
faiss_store_dir = "/home/Group_1/CREATE_VECTOR_STORE"
llama_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# === Load Models ===
encoder = SentenceTransformer(simcse_model_path)
index = faiss.read_index(os.path.join(faiss_store_dir, "q_index.faiss"))

with open(os.path.join(faiss_store_dir, "id2row.pkl"), "rb") as f:
    id2row = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
model = AutoModelForCausalLM.from_pretrained(llama_model_id, device_map="auto")

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1
)

# === Prompt Template ===
def build_prompt(context, question, belonging):
    return f"""
You are a highly knowledgeable Physics tutor helping students understand scientific concepts clearly and accurately.

Below is the context extracted from study materials, along with the topic or chapter they belong to.

---
Belongs to: {belonging}

Context:
{context}

Question:
{question}
---

Answer in a clear, concise, and explanatory manner:
""".strip()

# === Embedding & Retrieval ===
def embed_query(text):
    return encoder.encode(text, normalize_embeddings=True).reshape(1, -1)

def retrieve(query, k=3):
    vec = embed_query(query)
    scores, indices = index.search(vec, k)
    return [id2row[i] for i in indices[0]]

# === Main QA Function ===
def ask_question(question_text, k=3):
    top_chunks = retrieve(question_text, k)

    if isinstance(top_chunks[0], str):
        try:
            top_chunks = [json.loads(c) for c in top_chunks]
        except:
            context = "\n\n".join(top_chunks)
            belonging = "Unknown"
            prompt_text = build_prompt(context, question_text, belonging)
            response = generation_pipe(prompt_text)
            return response[0]["generated_text"].strip()

    context = "\n\n".join(chunk.get("text", "") for chunk in top_chunks)
    belonging = "; ".join(sorted(set(chunk.get("belonging", "Unknown") for chunk in top_chunks)))
    prompt_text = build_prompt(context, question_text, belonging)
    response = generation_pipe(prompt_text)
    
    return response[0]["generated_text"].strip()

# === Entry Point ===
if __name__ == "__main__":
    query = "Explain me newton's three law of motion"
    response = ask_question(query)
    print(response)
