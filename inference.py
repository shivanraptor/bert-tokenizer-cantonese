from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained('raptorkwok/bart-base-cantonese')
model = BartForConditionalGeneration.from_pretrained('raptorkwok/bart-base-cantonese', from_flax=True) # from_flax=True is required
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
output = text2text_generator('聽日就要返香港，我激動到[MASK]唔着', max_length=50, do_sample=False)
print(output[0]['generated_text'].replace(' ', ''))