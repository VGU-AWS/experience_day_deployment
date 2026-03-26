FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

WORKDIR /app

#RUN apt-get update && \
#    apt-get install -y python3.12-venv && \
#    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

#RUN python -m venv /opt/venv
#ENV PATH="/opt/venv/bin:$PATH"

#RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]
