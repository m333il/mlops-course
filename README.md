# Fashion Classification Project

## Описание проекта

Проект по автоматической классификации изображений одежды с использованием нейронных сетей. Система предназначена для интернет-магазинов одежды и позволяет автоматически категоризировать товары по загруженным фотографиям.

## Бизнес-цель

Автоматизация процесса категоризации товаров в интернет-магазине одежды для:
- Снижения времени обработки новых товаров
- Уменьшения ошибок категоризации
- Улучшения пользовательского опыта за счет точного поиска по категориям

## Целевые метрики

### Метрики качества модели
- **Accuracy** ≥ 80% - общая точность классификации
- **Top-3 Accuracy** ≥ 90% - правильная категория в топ-3 предсказаниях

### Метрики производительности (Production SLA)
- **Среднее время отклика** ≤ 200 мс
- **Пропускная способность** ≥ 100 запросов/сек
- **Доля неуспешных запросов** ≤ 1%
- **Использование памяти** ≤ 2 GB

## Набор данных

### DeepFashion Dataset
- **Источник**: [deepfashion2 datasest](https://github.com/switchablenorms/DeepFashion2)
- [HuggingFace link](https://huggingface.co/datasets/SaffalPoosh/deepFashion-with-masks)
- **Размер**: 40658 изображений для классификации
- **Категории**: 17 классов (dresses, shorts, cardigans и т.д.)


## Стек
  - Базовая модель: [ResNet18](https://huggingface.co/microsoft/resnet-18) предобученная на [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
  - Размер изображений: 224x224 пикселей
  - Изображения нормализуются (ImageNet statistics)


## План экспериментов
1. Baseline модель (ResNet-18)
2. Улучшение качества 
3. Оптимизация модели для продакшена 

---

## Хранение данных и моделей

Данные и модели хранятся в DVC remote на DagsHub: https://dagshub.com/m333il/mlops-course

---

## MLflow Tracking

### Настройка

Эксперименты трекаются через MLflow и DagsHub.

**Настройка environment variables:**

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/m333il/mlops-course.mlflow
export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

**Как получить токен DagsHub:**
1. Войдите на [dagshub.com](https://dagshub.com) (можно через github)
2. Перейдите в Your Settings -> Tokens
3. Создайте новый токен или используйте существующий

### Просмотр экспериментов

**Через DagsHub UI:**
- Откройте https://dagshub.com/m333il/mlops-course
- Перейдите во вкладку "Experiments"

**MLflow UI:**
- Откройте https://dagshub.com/m333il/mlops-course.mlflow/
- Во вкладке "Experiments" перейти в "fashion-classification"

### Что логируется

**Параметры:**
- `model_name` - название модели
- `dataset_name` - название датасета
- `learning_rate`, `weight_decay`, `batch_size`, `num_epochs`
- `seed` - random seed для воспроизводимости

**Метрики:**
- `train_loss` - loss на каждой эпохе
- `val_accuracy` - accuracy на валидации после каждой эпохи
- `final_accuracy` - финальная accuracy

**Артефакты:**
- `model/` - обученная модель (через mlflow.transformers)
- `config/base.yaml` - конфигурация эксперимента
- `metrics.json` - метрики в JSON формате
- `dvc.lock` - для связи с версией данных

**Теги:**
- `dvc_data_hash` - хеш dvc.lock для связи эксперимента с версией данных
- `git_commit` - git commit hash

---

## Воспроизводимость

### 1. Клонирование репозитория

```bash
git clone https://github.com/m333il/mlops-course.git
cd mlops-course
```

### 2. Настройка окружения

```bash
conda env create -f environment.yaml

conda activate mlops
```

### 3. Настройка credentials

**Для DVC (загрузка данных):**
```bash
export AWS_ACCESS_KEY_ID=<your_dagshub_token>
export AWS_SECRET_ACCESS_KEY=<your_dagshub_token>
```

**Для MLflow (трекинг экспериментов):**
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/m333il/mlops-course.mlflow
export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```
*Способ получения your_dagshub_token описан выше.*

### 4. Загрузка данных и модели

```bash
dvc pull
```

### 5. Воспроизведение пайплайна

```bash
# Запуск всего пайплайна
dvc repro

# Или отдельных стадий
dvc repro prepare
dvc repro train
dvc repro evaluate
```

---

## Запуск тестов

```bash
pytest tests/ -v
```

---

## Текущие результаты

- **Accuracy**: 0.7398
- **Модель**: ResNet-18 fine-tuned
- **Эпохи**: 5 
