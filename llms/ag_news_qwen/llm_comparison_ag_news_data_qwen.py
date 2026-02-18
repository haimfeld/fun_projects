import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import time

print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


train_df = pd.read_csv('/content/train.csv')[0:5000]
test_df = pd.read_csv('/content/test.csv')
display(train_df.head())

print(train_df.shape)
print(test_df.shape)

print(train_df.columns)

labels = {1:'World',2:'Sports',3:'Business',4:'Science/Tech'}
train_df['labels'] = train_df['Class Index'].map(labels)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

batch_size = 32

def build_prompt(row):
    return (
        "You are a helpful assistant that classifies news articles into one of four categories: World, Sports, Business, Science/Tech.\n"
        "Categories definitions:\n"
        "- Business: economy, stocks, companies, trade, finance.\n"
        "- Science/Tech: AI, research, technology, experiments, discoveries.\n"
        "- World: international news, diplomacy, politics.\n"
        "- Sports: sports events, scores, athletes.\n\n"
        "Examples:\n\n"
        "Text: Stocks fell sharply as investors reacted to Fed policy.\n"
        "Category: Business\n\n"
        "Text: Scientists unveil new method for quantum encryption.\n"
        "Category: Science/Tech\n\n"
        "Text: France signs new diplomatic trade deal with Canada.\n"
        "Category: World\n\n"
        "Text: Ronaldo scores twice as Manchester wins league match.\n"
        "Category: Sports\n\n"
        "Text: New solar panel technology increases energy efficiency.\n"
        "Category: Science/Tech\n\n"
        "Text: Central bank raises interest rates unexpectedly.\n"
        "Category: Business\n\n"
        "Now classify this article. Answer ONLY with the exact category name:\n"
        f"Title: {row['Title']}\n"
        f"Description: {row['Description']}\n"
        "Category:"
    )

prompts = [build_prompt(row) for _, row in train_df.iterrows()]

preds = []
start_time = time.time()

for i in tqdm(range(0, len(prompts), batch_size), desc="Qwen Inference"):
    batch_prompts = prompts[i:i+batch_size]

    messages = [
        [{"role": "system", "content": "You are Qwen, a helpful assistant."},
         {"role": "user", "content": p}]
        for p in batch_prompts
    ]

    text_batch = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    enc = tokenizer(text_batch, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=20
        )

    for input_ids, output_ids in zip(enc.input_ids, gen):
        output_ids = output_ids[len(input_ids):]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        preds.append(decoded)

train_df['pred_label'] = preds

total_time = time.time() - start_time
avg_time = total_time / len(train_df)
print(f"Finished {len(train_df)} samples in {total_time:.2f}s "
      f"({avg_time:.2f}s per sample, {1/avg_time:.2f} samples/sec)")


accuracy = (train_df["labels"] == train_df["pred_label"]).mean()
print("Accuracy:", accuracy)

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

y_true = train_df['Class Index'].map(labels)
y_pred = train_df['pred_label'].str.strip()

y_pred = y_pred.apply(lambda x: x if x in labels.values() else 'Unknown')

report = classification_report(y_true, y_pred, labels=list(labels.values()), zero_division=0)
print(report)


cm = confusion_matrix(y_true, y_pred, labels=list(labels.values()))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels.values(),
            yticklabels=labels.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
