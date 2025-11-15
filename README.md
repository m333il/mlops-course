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
1. Baseline модель
2. Улучшение качества
3. Оптимизация модели для продакшена


## Возпроизводимость
1. Install anaconda
2. run: conda env create -f environment.yaml
3. run: conda activate test-env
4. run: python train.py
5. Final accuracy: 0.7322, model saved to models/resnet-18
