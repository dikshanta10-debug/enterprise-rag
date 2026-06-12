import json
import requests
import statistics
import ollama
import time

API_URL = "http://localhost:8000/query"
OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"

with open("eval_questions.json") as f:
    questions = json.load(f)

rag_correct = 0
baseline_correct = 0
filter_rates = []
latencies = []
total_q = 0

def word_overlap_ratio(candidate, reference):
    """Fraction of reference words that appear in candidate."""
    candidate_words = set(candidate.lower().split())
    reference_words = set(reference.lower().split())
    if not reference_words:
        return 0.0
    common = candidate_words & reference_words
    return len(common) / len(reference_words)

client = ollama.Client(host=OLLAMA_HOST)

import time
print("Warming up – waiting 5 seconds for API to stabilise...")
time.sleep(5)

for i, item in enumerate(questions, 1):
    print(f"\n[{i}/{len(questions)}] Question: {item['question']}")
    
    # RAG
    resp = requests.post(API_URL, json={"question": item["question"]})
    if resp.status_code != 200:
        print(f"  RAG API error {resp.status_code}")
        continue
    data = resp.json()
    answer = data["answer"]
    chunks_retrieved = data["chunks_retrieved"]
    chunks_after = data["chunks_after_filter"]
    latency = data["retrieval_latency_ms"]
    latencies.append(latency)
    filter_rate = (chunks_retrieved - chunks_after) / chunks_retrieved if chunks_retrieved else 0
    filter_rates.append(filter_rate)
    print(f"  RAG: {answer[:120]}...")
    print(f"  Chunks: ret={chunks_retrieved}, after={chunks_after}, lat={latency:.1f}ms")
    if word_overlap_ratio(answer, item["relevant_text"]) >= 0.5:
        rag_correct += 1
        print("  RAG correct (token overlap >= 0.5)")
    else:
        print("  RAG incorrect")

    # Baseline
    try:
        baseline_resp = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": f"Answer concisely:\n{item['question']}"}]
        )
        baseline_answer = baseline_resp["message"]["content"]
        print(f"  Baseline: {baseline_answer[:120]}...")
        if word_overlap_ratio(baseline_answer, item["relevant_text"]) >= 0.5:
            baseline_correct += 1
            print("  Baseline correct")
        else:
            print("  Baseline incorrect")
    except Exception as e:
        print(f"  Baseline error: {e}")
        baseline_answer = ""
    
    total_q += 1
    time.sleep(0.2)

print("\n" + "="*50)
print("EVALUATION SUMMARY")
print(f"Total questions evaluated: {total_q}")
if latencies:
    print(f"Average retrieval latency: {statistics.mean(latencies):.1f} ms (min {min(latencies):.1f}, max {max(latencies):.1f})")
if filter_rates:
    print(f"Average guardrail filter rate: {statistics.mean(filter_rates)*100:.1f}%")
rag_acc = rag_correct / total_q * 100 if total_q else 0
baseline_acc = baseline_correct / total_q * 100 if total_q else 0
improvement_pp = rag_acc - baseline_acc
print(f"RAG accuracy: {rag_correct}/{total_q} ({rag_acc:.0f}%)")
print(f"Baseline accuracy: {baseline_correct}/{total_q} ({baseline_acc:.0f}%)")
print(f"Accuracy improvement: {improvement_pp:.0f} percentage points")