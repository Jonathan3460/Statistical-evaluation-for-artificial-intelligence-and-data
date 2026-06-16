from datasets import load_dataset
import pandas as pd
import os


PILOT_SIZE = 50
EXPERIMENT_SIZE = 400
TOTAL_QUESTIONS_NEEDED = PILOT_SIZE + EXPERIMENT_SIZE

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

if len(final_df) >= TOTAL_QUESTIONS_NEEDED:
    sampled_df = final_df.sample(n=TOTAL_QUESTIONS_NEEDED, random_state=42)
else:
    print(
        f"Warning: This subject only has {len(final_df)} questions. Using all of them.")
    sampled_df = final_df

pilot_df = sampled_df.iloc[0: PILOT_SIZE]
experiment_df = sampled_df.iloc[PILOT_SIZE: PILOT_SIZE + EXPERIMENT_SIZE]

os.makedirs("data", exist_ok=True)

pilot_filename = f"mmlu_pilot_data_n{PILOT_SIZE}.csv"
experiment_filename = f"mmlu_experiment_data_n{EXPERIMENT_SIZE}.csv"

pilot_df.to_csv(f"data/{pilot_filename}", index=False)
experiment_df.to_csv(f"data/{experiment_filename}", index=False)

print(f"Done! Saved {len(pilot_df)} questions to {pilot_filename}")
print(f"Done! Saved {len(experiment_df)} questions to {experiment_filename}")
