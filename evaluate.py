import os
import csv
import time
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
import rag_engine

# Load .env variables
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def evaluate_system(num_samples=10):
    if not GROQ_API_KEY:
        print(" Error: GROQ_API_KEY not found. Please set it in your .env file.")
        return

    # We use the best available model as the Judge
    try:
        judge_llm = rag_engine.get_groq_llm(GROQ_API_KEY)
    except Exception as e:
        print(f"Error: Failed to initialize Judge LLM: {e}")
        return

    csv_path = Path("data/csv/squad.csv")
    if not csv_path.exists():
        print(f" Dataset not found at {csv_path}")
        return

    # Read samples from the SQuAD dataset
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            samples.append(row)

    print(f" Starting evaluation on {len(samples)} samples...\n")
    print("=" * 60)
    
    total_score = 0
    processed = 0

    for idx, sample in enumerate(samples, 1):
        question = sample["question"]
        true_answer = sample["answer"]

        print(f"--- Sample {idx}/{len(samples)} ---")
        print(f" Question: {question}")
        
        # 1. Ask the RAG system
        start_time = time.time()
        try:
            rag_response = rag_engine.query(question)
            generated_answer = rag_response["answer"]
        except Exception as e:
            print(f" Error querying RAG engine: {e}\n")
            continue
        
        latency = time.time() - start_time

        # 2. Ask the Judge LLM to score the answer
        prompt = f"""You are an expert strict evaluator.
User Question: {question}
True Fact/Answer: {true_answer}
System Generated Answer: {generated_answer}

Score the System Generated Answer from 1 to 5 based on its factual accuracy and relevance compared to the True Fact.
A score of 1 means completely wrong, hallucinatory, or irrelevant.
A score of 5 means perfectly accurate and contains the true fact.
Only output a single integer from 1 to 5. Do not explain.
"""
        try:
            judge_response = judge_llm.complete(prompt)
            score_str = str(judge_response).strip()
            # Extract just the first digit found in case the LLM writes extra text
            score = int(''.join(filter(str.isdigit, score_str))[:1])
            if score == 0: 
                score = 1 # fallback if parsing fails
        except Exception as e:
            print(f" Error calling Judge LLM: {e}")
            score = 1 # Penalize if grading fails

        # Print the results
        print(f"o. True Answer: {true_answer}")
        if len(generated_answer) > 120:
            print(f"-> Generated:   {generated_answer[:120]}...")
        else:
            print(f"-> Generated:   {generated_answer}")
        
        # Colorize score based on value (ANSI escape codes)
        if score >= 4:
            score_display = f"\033[92m{score}/5\033[0m" # Green
        elif score == 3:
            score_display = f"\033[93m{score}/5\033[0m" # Yellow
        else:
            score_display = f"\033[91m{score}/5\033[0m" # Red

        print(f"   Score:       {score_display}  (Latency: {latency:.2f}s)\n")

        total_score += score
        processed += 1
        
        # Sleep for a second to avoid hitting free-tier Groq rate limits
        time.sleep(1)

    # 3. Calculate and display final metrics
    if processed > 0:
        avg_score = total_score / processed
        print("=" * 60)
        print(f"dY? FINAL EVALUATION SCORE: {avg_score:.2f} / 5.0")
        
        if avg_score >= 4.5:
            print("Verdict: EXCELLENT  - The model is highly accurate and faithful to the data.")
        elif avg_score >= 3.5:
            print("Verdict: GOOD dY' - The model is solid but might occasionally miss nuance.")
        elif avg_score >= 2.5:
            print("Verdict: FAIR  - The model is struggling. Check your chunk size or embeddings.")
        else:
            print("Verdict: POOR  - The model is failing to retrieve or answer correctly.")
    else:
        print("No samples were successfully processed.")

if __name__ == "__main__":
    # Test on the first 10 rows of the CSV
    evaluate_system(num_samples=10)
