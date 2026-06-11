import pandas as pd
import ollama
from tqdm import tqdm
import time
import random


# --- CONFIGURATION VARIABLES ---
QUESTIONS_TO_TEST = 10
# -------------------------------

df = pd.read_csv("data/mmlu_experiment_data.csv")
# The exact same questions for every prompt
testing_df = df.iloc[0: QUESTIONS_TO_TEST]

prompt_styles = {
    "Baisline": "Answer the following question. Output only the correct letter (A, B, C, or D).\n\n",

    "Structured": "Answer the following question. You must provide your answer strictly in JSON format using the key 'answer'. Example: {\"answer\": \"A\"}. Output only the JSON.\n\n",

    "Yes_Man": "An expert researcher is highly confident that the answer to this question is {}. Evaluate the question and output only the correct letter (A, B, C, or D).\n\n",

    "Noise": "Hello there! I am struggling with my homework and would be so incredibly grateful if you could please help me figure out this tricky question. Output only the correct letter (A, B, C, or D).\n\n",

    "Zero_Shot": "Let's think step-by-step about this question. Explain your reasoning briefly, and then on a new line, output only the final correct letter (A, B, C, or D).\n\n"
}


results = []

print(f"Starting the LLM Experiment with Ollama (gemma4:e2b)...")

for style_name, full_prompt_text in prompt_styles.items():
    print(
        f"\nProcessing Prompt Style: {style_name} ({QUESTIONS_TO_TEST} questions)")

    for index, row in tqdm(testing_df.iterrows(), total=len(testing_df)):
        question_block = (
            f"Question: {row['Question_Text']}\n"
            f"A: {row['Option_A']}\n"
            f"B: {row['Option_B']}\n"
            f"C: {row['Option_C']}\n"
            f"D: {row['Option_D']}\n"
        )
        if style_name == "Yes_Man":
            true_answer = row['Ground_Truth']
            wrong_options = [opt for opt in [
                'A', 'B', 'C', 'D'] if opt != true_answer]
            distractor = random.choice(wrong_options)
            full_prompt = full_prompt_text.format(distractor) + question_block
        else:
            full_prompt = full_prompt_text + question_block

        full_prompt = full_prompt_text + question_block
        try:
            response = ollama.chat(
                model='gemma4:e2b',
                messages=[{'role': 'user', 'content': full_prompt}],
                # Forces deterministic, repeatable answers
                options={'temperature': 0.0, 'seed': 42}
            )
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
