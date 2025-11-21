from flask import Flask, render_template, request, jsonify
import os
import re

try:
    import openai
except Exception:
    openai = None

app = Flask(__name__)

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def construct_prompt(processed_question: str) -> str:
    return f"You are a helpful assistant. Answer concisely. Question (preprocessed): {processed_question}"

def call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if openai is None or not api_key:
        raise RuntimeError("OpenAI SDK not available or OPENAI_API_KEY not set.")

    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2,
    )

    if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
        return resp["choices"][0].get("message", {}).get("content", "") or resp["choices"][0].get("text", "")
    return str(resp)

def local_fallback_answer(processed_question: str) -> str:
    if len(processed_question.split()) == 0:
        return "(Fallback) No question provided."
    return f"(Fallback) Could not reach LLM. Echo: {processed_question}. Set OPENAI_API_KEY for real answers."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        processed = preprocess(question)
        prompt = construct_prompt(processed)
        try:
            answer = call_openai(prompt)
            source = 'openai'
        except Exception:
            answer = local_fallback_answer(processed)
            source = 'fallback'
        return render_template('index.html', question=question, processed=processed, answer=answer, source=source)
    return render_template('index.html', question=None, processed=None, answer=None, source=None)

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    processed = preprocess(question)
    prompt = construct_prompt(processed)
    try:
        answer = call_openai(prompt)
        source = 'openai'
    except Exception:
        answer = local_fallback_answer(processed)
        source = 'fallback'
    return jsonify({
        "original_question": question,
        "processed_question": processed,
        "answer": answer,
        "source": source
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
