# sms-spam-filter

Классификатор SMS на спам и обычные сообщения. TF-IDF по словам и биграммам,
дальше наивный Байес или логистическая регрессия. Обучаются обе модели,
сохраняется та, у которой F1 на отложенной выборке выше.

Датасет: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(UCI), 5574 сообщения на английском, после удаления дублей 5171, спама 12.6%.
Скачивается сам при первом запуске.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python train.py                 # обе модели
python train.py --model logreg  # только одна
python predict.py "Free entry! Text WIN to 87121" "See you tomorrow"
```

`predict.py` без аргументов читает сообщения из stdin, по одному в строке.

## Результаты

На тестовых 20% (1035 сообщений, random_state=42):

| модель | precision | recall | F1 (спам) | accuracy |
|---|---|---|---|---|
| Naive Bayes | 0.976 | 0.924 | 0.949 | 0.987 |
| Logistic Regression | 0.961 | 0.947 | 0.954 | 0.988 |

Байес почти не трогает обычные сообщения (3 ложных срабатывания), но
пропускает больше спама. Логистическая регрессия с `class_weight="balanced"` ловит
больше спама ценой пары лишних срабатываний.

Перед векторизацией ссылки, суммы в валюте и длинные номера заменяются
на служебные слова `url`, `money`, `longnum`. Конкретный номер в обучающей
выборке почти никогда не повторяется, а сам факт номера в тексте хорошо
отличает спам.

## Тесты

```bash
pytest
```
