import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q

df = pd.read_csv("results/evaluation_matrix_experiment.csv")

data = df.drop(columns=["Question_ID"])

result = cochrans_q(data.values)

print("Q statistic:", result.statistic)
print("p-value:", result.pvalue)