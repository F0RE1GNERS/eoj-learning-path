FROM tensorflow/tensorflow:1.13.1
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ./entrypoint.sh
