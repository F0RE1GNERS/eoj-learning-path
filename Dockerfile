FROM tensorflow/tensorflow:1.13.1
ADD . /app
WORKDIR /app
RUN apk add --update --no-cache build-base && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    apk del build-base --purge
ENTRYPOINT ./entrypoint.sh
