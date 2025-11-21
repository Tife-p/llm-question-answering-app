#!/usr/bin/env python3
"""
LLM_QA_CLI.py
Python CLI application for a simple Q&A using an LLM API (OpenAI by default).
Usage:
  - Interactive: python LLM_QA_CLI.py
  - One-off question: python LLM_QA_CLI.py --question "What is NLP?"
Environment:
  - Set OPENAI_API_KEY if you want real LLM responses. If not set, a local fallback is used.
"""
import os
import argparse
import re
import json
from typing import List

# Optional import for OpenAI; program will still run without it.
try:
    import openai
except Exception:
    openai = None

def preprocess(text: str) -> str:
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def construct_prompt(processed_question: str) -> str:
    prompt = f\"\"\"You are a helpful question-answering assistant.
Answer concisely and clearly.
User question (preprocessed): \"{processed_question}\"
Provide a short answer and a one-sentence explanation where appropriate.\"\"\"
    return prompt

def call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if openai is None or not api_key:
        raise RuntimeError("OpenAI SDK not available or OPENAI_API_KEY not set.")
    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if hasattr(openai, 'ChatCompletion') else "gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=256,
            temperature=0.2,
        )
        # adapt to different response shapes
        if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
            return resp["choices"][0].get("message", {}).get("content", "") or resp["choices"][0].get("text", "")
        # fallback string
        return str(resp)
    except Exception as e:
        raise RuntimeError(f\"OpenAI request failed: {e}\") from e

def local_fallback_answer(processed_question: str) -> str:
    # Very simple heuristic fallback - echoes and gives a structured placeholder answer.
    # This ensures the CLI works even without an API key.
    tokens = processed_question.split()
    if "define" in tokens or "what" in tokens or "meaning" in tokens:
        return f\"(Fallback) I couldn't call an LLM. But here's a short fallback answer: '{processed_question}'. Try setting OPENAI_API_KEY for real answers.\"
    if len(tokens) == 0:
        return \"(Fallback) No question provided.\"
    # Generic template answer
    return f\"(Fallback) I couldn't call an LLM. Echo: '{processed_question}'. Set OPENAI_API_KEY to get a real LLM response.\"

def answer_question(question: str) -> dict:
    processed = preprocess(question)
    prompt = construct_prompt(processed)
    try:
        answer = call_openai(prompt)
    except Exception:
        answer = local_fallback_answer(processed)
    return {
        "original_question": question,
        "processed_question": processed,
        "prompt_sent": prompt,
        "answer": answer
    }

def main():
    parser = argparse.ArgumentParser(description=\"LLM Q&A CLI\")
    parser.add_argument("--question", "-q", type=str, help="Question to ask the LLM (if omitted, enters interactive mode)")
    args = parser.parse_args()

    if args.question:
        q = args.question
        result = answer_question(q)
        print(\"\\n--- LLM Q&A CLI Result ---\")
        print(f\"Original question: {result['original_question']}\")
        print(f\"Processed question: {result['processed_question']}\")
        print(\"\\nAnswer:\") 
        print(result['answer'])
    else:
        print(\"LLM Q&A CLI - interactive mode (type 'exit' or Ctrl+C to quit)\") 
        while True:
            try:
                q = input(\"\\nEnter your question: \").strip()
                if not q or q.lower() in (\"exit\",\"quit\"):
                    print(\"Goodbye!\") 
                    break
                result = answer_question(q)
                print(\"\\nProcessed question:\", result['processed_question'])
                print(\"\\nAnswer:\") 
                print(result['answer'])
            except KeyboardInterrupt:
                print(\"\\nInterrupted. Bye!\") 
                break

if __name__ == '__main__':
    main()
