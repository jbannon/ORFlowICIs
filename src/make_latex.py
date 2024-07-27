import pandas as pd

drug = 'Nivo'
tissue = 'SKCM'
df = pd.read_csv(f"../results/{drug}/{tissue}/go_table.txt",sep="\t")
df  = df[['goId', 'description']]
print(df.to_latex())