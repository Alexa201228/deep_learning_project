from transformers import MarianMTModel, MarianTokenizer

# Выбираем модель для перевода с английского на русский
model_name = 'Helsinki-NLP/opus-mt-en-ru'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate(text):
    # Токенизируем текст и переводим его
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=40)
    outputs = model.generate(inputs, max_length=40)

    # Декодируем результат и возвращаем перевод
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text
