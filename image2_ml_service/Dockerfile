FROM python:3.12-alpine

#ARGS are quite useful, they are used to indicate metadata, version/requirements build date

ARG BUILD_DATE="18/02/2026"
ARG VERSION=1.0.0
ARG REQUIREMENTS=requirements.txt


LABEL maintainer="Lefki Meidi <m.lef3105@gmail.com>"
LABEL version=$VERSION
LABEL build_date=$BUILD_DAT
LABEL description="I overcomplicate my tp, just to piss off ustaz"


WORKDIR /app

COPY . /app/

RUN py -m pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python","main.py"]