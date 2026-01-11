FROM python:3.12-slim

WORKDIR /app

# python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# source code
COPY src/ ./src/

# trained model (run `dvc pull` before build)
COPY models/resnet-18/ ./models/resnet-18/

# directories for input/output
RUN mkdir -p /data/input /data/output

ENTRYPOINT ["python", "-m", "src.predict"]

CMD ["--input_path", "/data/input", "--output_path", "/data/output/preds.csv"]
