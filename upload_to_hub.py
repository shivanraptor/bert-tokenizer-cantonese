from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('vocab-bart-base-cantonese.txt')
tokenizer.push_to_hub('raptorkwok/bart-base-cantonese')

# added the FlaxBartModel to commit
model = FlaxBartModel.from_pretrained('fnlp/bart-base-chinese', from_pt=True)
model.push_to_hub('raptorkwok/bart-base-cantonese', private=True)