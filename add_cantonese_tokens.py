from collections import Counter
#from random import Random
from transformers import BertTokenizerFast

from lib import token_to_token_id as vocab_old, is_unused, is_cjkv

#rng = Random(42)
#sentences = load_lihkg()
#sentences = rng.choices(sentences, k=524288)

#tokenizer_old = BertTokenizerFast.from_pretrained('fnlp/bart-base-chinese') # not used
# tokenizer_new = tokenizer_old.train_new_from_iterator(sentences, 2048, length=len(sentences))
sentences = pickle.load(open("data/wordsegs.pickle", "rb"))
print("Number of Word Segments:", len(sentences))
# Output: Number of Word Segments: 462064472

########

vocab_new = set()

with open('outputs/vocab_mapping.txt', encoding='utf-8') as f:
    for line in f:
        token, token_id = line.rstrip('\n').rsplit(' ', 1)
        if is_unused(token):
            continue
        vocab_new.add(token)

########

def check_valid_token(t):
    return (is_cjkv2(t) and t not in vocab_new)

counter = Counter(filter(check_valid_token, sentences[0]))

cjkv_new = set((token for token, _ in counter.most_common(15000))) # change from 150 to 15000

########
# save the Counter to pickle
with open('outputs/counter.pickle', 'wb') as c:
    pickle.dump(counter, c)

########

with open('data/yue.txt', encoding='utf-8') as f:
    text = f.read()

for c in text:
    if is_cjkv(c) and c not in vocab_new:
        cjkv_new.add(c)

########

cs = set()
for c in cjkv_new:
    if c not in vocab_old:
        cs.add(c)

########

with open('outputs/add_token.txt', 'w', encoding='utf-8') as f:
    print('\n'.join(sorted(cs)), file=f)
