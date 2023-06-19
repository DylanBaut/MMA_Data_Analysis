import pandas as pd
scores_df = pd.read_csv('scorerData.csv')
print(len(scores_df)) 
for rowtuple in scores_df.itertuples():
    row= list(rowtuple)[1:]
    print(row[1])
