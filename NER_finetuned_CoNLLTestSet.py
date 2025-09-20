# !pip install transformers datasets seqeval -q

"""
do not forget to add all used libraries to the requirements.txt
"""
import os, time, torch
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import f1_score, accuracy_score

# 1️⃣ Load your local CoNLL-2003 data
"""
be aware with the potential issues of the dataset https://github.com/synalp/NER/issues
"""

data_dir = "/content/conll2003"


def read_conll(path):
    tokens, tags = [], []
    all_tokens, all_tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    all_tokens.append(tokens)
                    all_tags.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split(" ")
                tokens.append(splits[0])
                tags.append(splits[-1])
    return {"tokens": all_tokens, "ner_tags": all_tags}


"""
good!
"""

train = read_conll(os.path.join(data_dir, "eng.train"))
valid = read_conll(os.path.join(data_dir, "eng.testa"))
test = read_conll(os.path.join(data_dir, "eng.testb"))

"""
good!
"""

dataset = DatasetDict({
    "train": Dataset.from_dict(train),
    "validation": Dataset.from_dict(valid),
    "test": Dataset.from_dict(test),
})

print(dataset)
print("Sample:", dataset["train"][0])

# 2️⃣ Get label list from training data
unique_tags = set(tag for doc in dataset["train"]["ner_tags"] for tag in doc)
label_list = sorted(list(unique_tags))
print("Labels:", label_list)


# 3️⃣ Convert Hugging Face spans → BIO
def hf_preds_to_bio(tokens, ner_results):
    """
    make sure you understand the format of the data,
    it seems to me it does not follow the standard BIO-format,
    but rather IOB, which is:


    ['Waterville', 'WA', '5.05', 'up', '.02', '---', '---', '4.6200', 'unc']
    ['I-LOC', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

    this is an example from the train data,
    you can see B-LOC is used to separate the last adjacent LOC tag

    your BIO construction now risks having many false positives, which are actually true positives

    GOLD: ['I-LOC', 'O', 'I-LOC', 'I-LOC', 'I-LOC', 'O']
    PREDICTED: ['B-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O']
    LABELS:    [fp, tp, fp, tp, tp, tp]

    for example, you have two fps which are actually tps if you change the format

    refer to https://github.com/synalp/NER/issues/1
    """

    pred_tags = ["O"] * len(tokens)
    for ent in ner_results:
        ent_type = ent["entity_group"]
        start, end = ent["start"], ent["end"]
        char_pos = 0
        inside = False
        for i, tok in enumerate(tokens):
            char_pos_end = char_pos + len(tok)
            if (start < char_pos_end) and (end > char_pos):
                if not inside:
                    pred_tags[i] = f"B-{ent_type}"
                    inside = True
                else:
                    pred_tags[i] = f"I-{ent_type}"
            char_pos = char_pos_end + 1
    return pred_tags


# 4️⃣ Evaluation function
def evaluate_model(model_name, dataset, n_samples=None):
    print(f"\n🚀 Evaluating {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    # good, you can experiment with different aggregations like max, average, etc.
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

    true_labels, pred_labels = [], []

    # good
    test_split = dataset["test"] if n_samples is None else dataset["test"].select(range(n_samples))
    start_time = time.time()

    for example in test_split:
        tokens = example["tokens"]
        gold = example["ner_tags"]
        true_labels.append(gold)

        preds = ner_pipe(" ".join(tokens))
        """
            please read the comments in hf_preds_to_bio
        """
        bio_preds = hf_preds_to_bio(tokens, preds)
        pred_labels.append(bio_preds)

    end_time = time.time()
    total_time = end_time - start_time

    f1 = f1_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)

    print(f"⏱ Inference time: {total_time:.2f}s on {len(test_split)} samples")
    print(f"📊 F1 score: {f1:.4f}")
    print(f"✅ Accuracy: {acc:.4f}")


"""
good!
"""
# 5️⃣ Compare DistilBERT vs BERT
evaluate_model("elastic/distilbert-base-cased-finetuned-conll03-english", dataset, n_samples=200)
evaluate_model("dslim/bert-base-NER", dataset, n_samples=200)

"""
if you want to show your output, better use a notebook
"""

# output
# DatasetDict({
#     train: Dataset({
#         features: ['tokens', 'ner_tags'],
#         num_rows: 14041
#     })
#     validation: Dataset({
#         features: ['tokens', 'ner_tags'],
#         num_rows: 3250
#     })
#     test: Dataset({
#         features: ['tokens', 'ner_tags'],
#         num_rows: 3453
#     })
# })
# Sample: {'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], 'ner_tags': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']}
# Labels: ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']

# 🚀 Evaluating elastic/distilbert-base-cased-finetuned-conll03-english
# Device set to use cpu
# ⏱ Inference time: 13.35s on 200 samples
# 📊 F1 score: 0.9387
# ✅ Accuracy: 0.9864

# 🚀 Evaluating dslim/bert-base-NER
# Some weights of the model checkpoint at dslim/bert-base-NER were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
# - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# Device set to use cpu
# ⏱ Inference time: 25.66s on 200 samples
# 📊 F1 score: 0.9050
# ✅ Accuracy: 0.9790
