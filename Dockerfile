FROM ufoym/deepo:pytorch-py36-cpu
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ./entrypoint.sh