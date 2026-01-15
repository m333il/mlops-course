# Мониторинг

### Сборка

```bash
dvc pull models/resnet-18
python -m serve.export_model
./serve/build_mar.sh
./scripts/build_serve.sh
```

### Запуск сервиса

```bash
docker run -d \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    --name fashion-server \
    fashion-serve:v1
```

### Проверка работы

**Статус сервиса:**
```bash
curl http://localhost:8081/models
```

**Метрики:**
```bash
curl http://localhost:8082/metrics
```

**Отправка запроса на классификацию:**
```bash
curl -X POST http://localhost:8080/predictions/fashion_classifier -T path/to/image.jpg
```

**Просмотр логов:**
```bash
docker logs fashion-server
```

### Остановка сервиса

```bash
docker stop fashion-server && docker rm fashion-server
```