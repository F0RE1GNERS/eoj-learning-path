FROM python3.6-alpine3.6
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ./entrypoint.sh
