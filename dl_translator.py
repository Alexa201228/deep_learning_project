from transformers import MarianMTModel, MarianTokenizer

# Выбираем модель для перевода с английского на русский
model_name = 'Helsinki-NLP/opus-mt-en-ru'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def translate(text: str):
    # Токенизируем текст и переводим его
    translated_text_sentences = []
    text = text.replace("\"", "<br/>")
    original_text = text.split("\n")
    for sentence in original_text:
        inputs = tokenizer.encode(sentence, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs)

        # Декодируем результат и возвращаем перевод
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_text_sentences.append(translated_text)
    return " ".join(translated_text_sentences)
