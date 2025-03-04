FROM akraradets/ait-ml-base:2023

RUN apt update && apt upgrade -y

RUN pip3 install --upgrade pip
RUN pip3 install mlflow

# Clean apt 
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow
# CMD mlflow server -h 0.0.0.0 -w 2
CMD mlflow server -h 0.0.0.0 --port 5001

# CMD mlflow server -h 0.0.0.0 -p 5000 --gunicorn-opts "--timeout 120"