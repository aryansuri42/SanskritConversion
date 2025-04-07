import pandas as pd

file_path = r"A:\Projects\Sanskrit Text Conversion\Sanskrit-Text-Conversion\Gita-data.csv"
csvFile = pd.read_csv(file_path)
csvFile = csvFile.drop("id", axis="columns")
df = pd.read_parquet("hf://datasets/VinitT/Sanskrit-Llama/data/train-00000-of-00001.parquet")
print(df.head())
desired_data = df.drop("instruction", axis='columns')
desired_data = pd.concat([desired_data, csvFile],ignore_index=True)
desired_data = desired_data.drop_duplicates()
desired_data = desired_data.dropna()
# print(desired_data.head(5))

random_sample = desired_data.sample(frac=1)

training_size = int(len(desired_data)*(80/100))
test_size = len(desired_data) - training_size
train_data = random_sample[:training_size]
test_data = random_sample[training_size:]


with open(r"Sanskrit-Text-Conversion\Data Files\TestEnglish.txt", "w", encoding="utf-8") as f:
    for i in test_data['output']:
        sent = i+"\n"
        f.write(sent)
    f.close()
print("done")

with open(r"Sanskrit-Text-Conversion\Data Files\TrainEnglish.txt", "w", encoding="utf-8") as f:
    for i in train_data['output']:
        sent = i+"\n"
        f.write(sent)
    f.close()
print("done")

with open(r"Sanskrit-Text-Conversion\Data Files\TestSanskrit.txt", "w", encoding="utf-8") as f:
    for i in test_data['input']:
        sent = i+"\n"
        f.write(sent)
    f.close()
print("done")

with open(r"Sanskrit-Text-Conversion\Data Files\TrainSanskrit.txt", "w", encoding="utf-8") as f:
    for i in train_data['input']:
        sent = i+"\n"
        f.write(sent)
    f.close()
print("done")
