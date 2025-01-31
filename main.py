import pandas as pd
import torch
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification

# Завантаження даних
df = pd.read_csv('Dropbox.csv')
df = df.sample(100)  # Вибірка 100 записів для аналізу

# Завантаження токенізатора і моделі
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Створення pipeline для аналізу настроїв
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Збір текстів з DataFrame
texts = list(df['content'].values)

# Отримання результатів
results = nlp(texts)

# Виведення результатів
for text, result, score in zip(texts, results, df['score'].values):
    print('Text:', text)
    print('Result:', result)
    print('Score:', score)

df['sentiment'] = [r['label'] for r in results]
print(df)