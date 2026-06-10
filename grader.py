import pandas as pd
import re


def extract_final_answer(text):
    """
    Attempts to extract the final A, B, C, or D from a messy AI response.
    """
    if pd.isna(text):
        return "ERROR"

    text = str(text).strip()

    json_match = re.search(r'"answer"\s*:\s*"([ABCD])"', text, re.IGNORECASE)
    if json_match:
        return json_match.group(1).upper()

    matches = re.findall(r'\b([ABCD])\b', text)
    if matches:
        return matches[-1].upper()

    return "UNKNOWN"


df = pd.read_csv("results/raw_ai_results.csv")

df['Extracted_Answer'] = df['Raw_AI_Response'].apply(extract_final_answer)

df['Is_Correct'] = (df['Extracted_Answer'] == df['Ground_Truth']).astype(int)

print("Extraction Preview (Checking for parsing errors):")
print(df[['Prompt_Style', 'Ground_Truth', 'Extracted_Answer', 'Is_Correct']].head(10))

eval_matrix = df.pivot(
    index='Question_ID', columns='Prompt_Style', values='Is_Correct')

eval_matrix.to_csv("results/evaluation_matrix.csv")
