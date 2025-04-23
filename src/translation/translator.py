from transformers import MBartForConditionalGeneration, MBart50Tokenizer

tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_fast=False)

tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "ur_PK"

model_path = "./data/models/fine_tuned_mbart50"
fine_tuned_model = MBartForConditionalGeneration.from_pretrained(model_path)


def translate_english_to_urdu(text):
    encoded_input = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)

    generated_tokens = fine_tuned_model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id["ur_PK"], 
        max_length=128
    )

    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text