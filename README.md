# Распознавание овощей по фото — Часть 1 + Трек B

Классификатор 6 овощей (Capsicum, Carrot, Cucumber, Potato, Raddish, Squash):
эмбеддинги замороженной ResNet18 (ImageNet) → `StandardScaler` → `LogisticRegression(class_weight='balanced')`
→ (при необходимости) калибровка `CalibratedClassifierCV(method='sigmoid')`.

Главный акцент — честная оценка: фото — это кадры из 19 видео, поэтому все метрики считаются out-of-fold
на **разбиении по видео** (`StratifiedGroupKFold`, группа = видео + почти-дубликаты), а не на готовом `final_dataset/`
(в нём 90% тестовых фото — копии фото из train).

## Структура

```
vegetables_trackB.ipynb   разбор данных (Часть 1) + все эксперименты Трека B
predict.py                инференс: папка с фото -> preds.csv
weights/
  backbone_resnet18.pt    веса ResNet18 без головы (чтобы не качать из интернета)
  classifier.joblib       обученный классификатор
figures/                  графики для отчёта
results.json              все цифры из ноутбука
requirements.txt          для predict.py (CPU)
requirements-train.txt    для ноутбука
REPORT.md                 отчёт (до 2 страниц)
```

## Запуск predict.py (CPU, без интернета)

```bash
pip install -r requirements.txt
python predict.py --input path/to/folder --output preds.csv
```

Необязательно: `--threshold 0.83` — фото с уверенностью ниже порога получат метку `unknown`
(обоснование порога 5/6 — в ноутбуке, раздел «Бонус»).

Формат `preds.csv`: `filename,label,confidence`. Скрипт печатает время на одно фото на CPU.

## Воспроизведение обучения

1. Открыть `vegetables_trackB.ipynb` в Google Colab, Runtime → T4 GPU.
2. Runtime → Run all. Датасет скачивается автоматически через `kagglehub`
   (`preethi866/vegetable-dataset`).
3. Ноутбук сохраняет веса, `requirements.txt` с точными версиями, графики и `results.json`
   в `/content/repo/` и скачивает `repo.zip`.

Все случайности зафиксированы (`SEED = 42`). Версии библиотек — в `requirements*.txt`
(генерируются ноутбуком из окружения, где он был запущен).
