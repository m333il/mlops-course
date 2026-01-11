## TorchServe

### Требования

- Docker
- Модель должна быть скачана локально (`dvc pull models/resnet-18`)
- torch-model-archiver (`pip install torch-model-archiver`)

### Подготовка и сборка

```bash
# Скачайте модель
dvc pull models/resnet-18

# Экспортируйте модель в TorchScript формат
python -m serve.export_model

# Создайте .mar архив
./serve/build_mar.sh

# Соберите Docker-образ
docker build -f Dockerfile.serve -t fashion-serve:v1 .
```

### Запуск сервиса

```bash
# Запуск контейнера
docker run -d -p 8080:8080 -p 8081:8081 --name fashion-server fashion-serve:v1

# Проверка статуса
docker logs fashion-server

# Остановка
docker stop fashion-server && docker rm fashion-server
```

### API Endpoints

| Endpoint | Port | Описание |
|----------|------|----------|
| `/predictions/fashion_classifier` | 8080 | Inference - классификация изображения |
| `/models` | 8081 | Management - список моделей |

### Примеры запросов

**Список зарегистрированных моделей:**
```bash
curl http://localhost:8081/models
```

**Классификация изображения:**
```bash
curl -X POST http://localhost:8080/predictions/fashion_classifier -T path/to/image.jpg
```

**Пример ответа:**
```json
{
  "label": "Dresses",
  "class_id": 3,
  "confidence": 0.9234
}
```


### Форматы данных

**Входные данные:**
- Файл изображения
- Поддерживаемые форматы: JPEG, PNG, WebP, BMP

**Выходные данные (JSON):**
```json
{
  "label": str,      // Название категории одежды
  "class_id": int,  // ID класса (0-16)
  "confidence": float   // Уверенность предсказания (0-1)
}
```


### Конфигурация

Параметры сервиса настраиваются в `serve/config.properties`:

```properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
default_workers_per_model=1
default_response_timeout=120
```

