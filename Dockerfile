FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (this layer is cached when only the code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and the dataset
COPY data_prep.py train.py predict.py ./
COPY data/ data/

# Train the model while building the image, so the model always matches the
# library versions inside the container (no pickle version problems)
RUN python train.py

EXPOSE 9696

# gunicorn is a production web server (Flask's built-in server is for development only)
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]