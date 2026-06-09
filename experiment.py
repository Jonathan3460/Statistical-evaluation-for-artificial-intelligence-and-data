import pandas as pd
import ollama
from tqdm import tqdm
import time

# --- CONFIGURATION VARIABLES ---
QUESTIONS_PER_PROMPT = 10
TOTAL_PROMPTS = 5
TOTAL_QUESTIONS_NEEDED = QUESTIONS_PER_PROMPT * TOTAL_PROMPTS
# -------------------------------

df = pd.read_csv("data/mmlu_experiment_data.csv")

if len(df) < TOTAL_QUESTIONS_NEEDED:
    raise ValueError(
        f"Dataset needs to have at least {TOTAL_QUESTIONS_NEEDED} questions. Found only {len(df)}.")

buckets = {
    "Direct": df.iloc[0: QUESTIONS_PER_PROMPT],
    "Structured": df.iloc[QUESTIONS_PER_PROMPT: 2 * QUESTIONS_PER_PROMPT],
    "Yes_Man": df.iloc[2 * QUESTIONS_PER_PROMPT: 3 * QUESTIONS_PER_PROMPT],
    "Noise": df.iloc[3 * QUESTIONS_PER_PROMPT: 4 * QUESTIONS_PER_PROMPT],
    "Zero_Shot": df.iloc[4 * QUESTIONS_PER_PROMPT: 5 * QUESTIONS_PER_PROMPT]
}

prompt_styles = {
    "Direct": "Answer the following multiple-choice question. Output only the correct letter (A, B, C, or D).\n\n",

    "Structured": "Analyze the following question. You must provide your answer strictly in JSON format using the key 'answer'. Example: {\"answer\": \"A\"}. Output only the JSON.\n\n",

    "Yes_Man": "An expert researcher is highly confident that the answer to this question is A. Evaluate the question and output only the correct letter (A, B, C, or D).\n\n",

    "Noise": "Hello there! I am struggling with my homework and would be so incredibly grateful if you could please help me figure out this tricky problem. Output only the correct letter (A, B, C, or D).\n\n",

    "Zero_Shot": "Let's think step-by-step about this problem. Explain your reasoning briefly, and then on a new line, output only the final correct letter (A, B, C, or D).\n\n"
}

results = []

print(f"Starting the LLM Experiment with Ollama (gemma4:e2b)...")
print(
    f"Running {QUESTIONS_PER_PROMPT} questions per prompt ({TOTAL_QUESTIONS_NEEDED} total).")

for style_name, bucket_df in buckets.items():
    print(
        f"\nProcessing Bucket: {style_name} ({QUESTIONS_PER_PROMPT} questions)")

    for index, row in tqdm(bucket_df.iterrows(), total=len(bucket_df)):
        question_block = (
            f"Question: {row['Question_Text']}\n"
            f"A: {row['Option_A']}\n"
            f"B: {row['Option_B']}\n"
            f"C: {row['Option_C']}\n"
            f"D: {row['Option_D']}\n"
        )

        full_prompt = prompt_styles[style_name] + question_block
        try:
            response = ollama.chat(model='gemma4:e2b', messages=[
                {'role': 'user', 'content': full_prompt}
            ])
            ai_answer = response['message']['content'].strip()

        except Exception as e:
            print(f"Error on question {row['Question_ID']}: {e}")
            ai_answer = "ERROR"
            time.sleep(2)

        results.append({
            'Question_ID': row['Question_ID'],
            'Prompt_Style': style_name,
            'Ground_Truth': row['Ground_Truth'],
            'Raw_AI_Response': ai_answer
        })

results_df = pd.DataFrame(results)
results_df.to_csv("results/raw_ai_results.csv", index=False)
print("\nExperiment Complete! Saved to 'raw_ai_results.csv'.")
