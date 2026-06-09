from datasets import load_dataset
import pandas as pd


print("Downloading MMLU dataset from Hugging Face...")
dataset = load_dataset("cais/mmlu", "all", split="test")
df = dataset.to_pandas()

print(f"Successfully loaded {len(df)} questions. Formatting data...")

letter_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
df['Ground_Truth'] = df['answer'].map(letter_mapping)

df['Option_A'] = df['choices'].apply(lambda x: x[0])
df['Option_B'] = df['choices'].apply(lambda x: x[1])
df['Option_C'] = df['choices'].apply(lambda x: x[2])
df['Option_D'] = df['choices'].apply(lambda x: x[3])

df = df.rename(columns={'question': 'Question_Text'})
df['Question_ID'] = ['Q' + str(i).zfill(3) for i in range(1, len(df) + 1)]

final_df = df[['Question_ID', 'Question_Text', 'Option_A',
               'Option_B', 'Option_C', 'Option_D', 'Ground_Truth']]

TOTAL_QUESTIONS_NEEDED = 500

if len(final_df) >= TOTAL_QUESTIONS_NEEDED:
    sampled_df = final_df.sample(n=TOTAL_QUESTIONS_NEEDED, random_state=42)
else:
    print(
        f"Warning: This subject only has {len(final_df)} questions. Using all of them.")
    sampled_df = final_df

csv_filename = "mmlu_experiment_data.csv"
sampled_df.to_csv("data/" + csv_filename, index=False)
print(f"Done! Saved {len(sampled_df)} formatted questions to {csv_filename}")
