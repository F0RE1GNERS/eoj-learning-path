FROM python:3.6-alpine3.6
ADD . /app
WORKDIR /app
RUN apk add --update --no-cache build-base && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    apk del build-base --purge
ENTRYPOINT ./entrypoint.sh
