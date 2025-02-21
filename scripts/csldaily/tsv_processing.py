import pickle
import pandas as pd

with open(
    "csldaily/sentence_label/csl2020ct_v2.pkl", "rb"
) as f:
    data = pickle.load(f)
df = pd.DataFrame(data["info"])

df_split = pd.read_csv(
    "csldaily/sentence_label/split_1.txt", sep="|"
)

dict_split = df_split.set_index("name").to_dict()["split"]

df["split"] = df["name"].replace(dict_split)

df = df[df.split.isin(["train", "dev", "test"])]

df.to_csv("data/csldaily/data.tsv", sep="\t", index=False)
