from datasets import load_dataset
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# val 0.8705
# test 0.8612

model_path = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
model.cuda()

base_path = "./datasets"

train_parquet = os.path.join(base_path, "cbt/CN/train-00000-of-00001.parquet")
val_parquet = os.path.join(base_path, "cbt/CN/validation-00000-of-00001.parquet")
test_parquet = os.path.join(base_path, "cbt/CN/test-00000-of-00001.parquet")

cbt = load_dataset("parquet", data_files={'train': train_parquet, 'val': val_parquet, 'test': test_parquet})

def calculate_log_likelihood(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        log_likelihood = -loss.item() * input_ids.size(1)
    return log_likelihood


def sliding_window(sequence, max_length, stride):
    for i in range(0, len(sequence), stride):
        yield sequence[i:i + max_length]


max_length = 1024
stride = 512
def predict_blank(context, candidates):
    max_log_likelihood = float('-inf')
    best_candidate = None

    for candidate in candidates:
        input_text = context.replace('XXXXX', candidate)
        input_ids = tokenizer.encode(input_text, return_tensors='pt')[0]

        input_ids = input_ids.cuda()

        if input_ids.shape[0] > max_length:
            log_likelihoods = []
            for window in sliding_window(input_ids, max_length, stride):
                window_input_ids = window.unsqueeze(0)
                log_likelihood = calculate_log_likelihood(window_input_ids, model)
                log_likelihoods.append(log_likelihood)

            total_log_likelihood = sum(log_likelihoods)
        else:
            input_ids = input_ids.unsqueeze(0)
            total_log_likelihood = calculate_log_likelihood(input_ids, model)

        if total_log_likelihood > max_log_likelihood:
            max_log_likelihood = total_log_likelihood
            best_candidate = candidate

    return best_candidate

cbt_split = cbt['val']

result = []
logs = []

for i in range(len(cbt_split)):
    ctx = cbt_split[i]['sentences']
    qst = cbt_split[i]['question']
    ans = cbt_split[i]['answer']
    opt = cbt_split[i]['options']

    i0 = "".join(ctx) + ' ' + qst

    res = predict_blank(i0, opt)

    if res == ans:
        result.append(1.0)
    else:
        result.append(0.0)

    logs.append((res, ans))
    print(sum(result) / len(result))


print(sum(result) / len(result))
