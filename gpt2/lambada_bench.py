import torch
import json
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
model.cuda()

with open('../data/lambada_test.jsonl', 'r') as fid:
    ds_raw = fid.read()

ds = ds_raw.strip().split('\n')

def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    text = text.replace('’', '\'')
    text = text.replace('‘', '\'')
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('…', '...')
    return text.strip()

lines = []
for line in ds:
    obj = json.loads(line)
    text = obj['text'].strip()

    text = preprocess(text)
    splited = text.split(' ')

    input_text = ' '.join(splited[:-1])
    input_label = splited[-1]

    lines.append([input_text, input_label])

ds = lines
data_loader = DataLoader(ds, batch_size=4, shuffle=False)

stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

stop_word_ids = []
for stop_word in stop_words:
    stop_word_id = tokenizer.encode(stop_word)
    stop_word_ids.extend(stop_word_id)

def score_batch(batch):
    batch_encoded = []

    encoded = tokenizer(batch[0], padding=True, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    with torch.no_grad():
        for _ in range(6):
            generated = [input_ids[i, :attention_mask[i].sum()] for i in range(input_ids.shape[0])]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            masked_logits = logits * attention_mask.unsqueeze(-1)
            non_padded_logits = [masked_logits[i, :attention_mask[i].sum(), :] for i in range(masked_logits.size(0))]

            last_index = []
            for i in range(masked_logits.size(0)):
                non_padded_logit = non_padded_logits[i]
                _, line_encoded_candidates = torch.topk(non_padded_logit[-1,:], k=128, dim=0)

                flag = True
                for j in range(line_encoded_candidates.shape[0]):
                    d = tokenizer.decode(line_encoded_candidates[j]).strip().lower()
                    if d not in stop_words:
                        last_index.append(line_encoded_candidates[j])
                        flag = False
                        break

                if flag:
                    last_index.append(line_encoded_candidates[0])

            last_argmax_logits = last_index
            generated = [torch.cat([generated[i], last_index[i].unsqueeze(0)]) for i in range(len(last_index))]

            new_attention_masks = []
            for i in range(attention_mask.shape[0]):
                if attention_mask[i].sum() == attention_mask[i].shape[0]:
                    new_attention_masks.append(torch.ones(attention_mask[i].shape[0] + 1).to(attention_mask[i]))
                else:
                    zero_pad = torch.zeros([1]).to(attention_mask)
                    new_attention_mask = torch.cat([attention_mask[i], zero_pad])
                    new_attention_mask[attention_mask[i].sum()] = 1.0
                    new_attention_masks.append(new_attention_mask)

            new_attention_masks = torch.stack(new_attention_masks, dim=0)

            pad_size = new_attention_masks.shape[1]
            new_pad_inputs = []
            for i in range(len(generated)):
                new_pad_input = torch.ones([pad_size]).to(input_ids) * 50256
                new_pad_input[:generated[i].shape[0]] = generated[i]
                new_pad_inputs.append(new_pad_input)

            new_pad_inputs = torch.stack(new_pad_inputs, dim=0)

            input_ids =  new_pad_inputs
            attention_mask = new_attention_masks

    decodes = []
    for ge in generated:
        decoded = tokenizer.decode(ge)
        decodes.append(decoded)


    tfs = []
    for i in range(len(decodes)):
        length = len(batch[0][i])

        decoded = decodes[i][length:]
        decoded = decoded.strip().split(' ')

        pd = decoded[0]
        gt = batch[1][i]

        tfs.append([pd, gt])

    return tfs

acc = []
writer = open('res.txt', 'w')
for i, batch in enumerate(data_loader):
    results = score_batch(batch)

    for index, res in enumerate(results):
        pd, gt = res

        length = len(gt)
        pd = pd[:length].lower()
        gt = gt.lower()

        strs = [pd, gt]
        if pd == gt:
            acc.append(1.0)
            strs.append('1')
        else:
            acc.append(0.0)
            strs.append('0')

        print("Acc: {}".format(sum(acc) / len(acc)))
        writer.write('|'.join(strs) + '\n')
    writer.flush()

writer.close()
