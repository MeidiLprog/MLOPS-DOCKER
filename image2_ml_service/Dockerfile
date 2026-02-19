FROM python:3.12-alpine 
#LAYER 1 OS+PYTHON

#ARGS are quite useful, they are used to indicate metadata, version/requirements build date and only exist during 
#docker build

ARG BUILD_DATE="2026-02-18"
ARG VERSION=1.0.0
ARG REQUIREMENTS=requirements.txt

#MUST USE, very useful for CI/CD 
LABEL org.opencontainers.authors="Lefki Meidi <m.lef3105@gmail.com>"
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.created=$BUILD_DAT
LABEL org.opencontainers.source="https://github.com/MeidiLprog/MLOPS-DOCKER.git"
LABEL org.opencontainers.image.description="I overcomplicate my tp, just to piss off ustaz"

#prevent python from creating useless files
#pycache no main.cython.pyc FIRST LINE
#NO DELAYS for the buferrng
#NO CACHE
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONBUFFERED=1 \
    PIP_NO_CACHE=1

#apk <=> APT? NO CACHE(explicit) build-base make/gcc etc libffi-dev = dependancies
RUN apk add --no-cache build-base libffi-dev

#Where everything shall be done
WORKDIR /app

COPY ${REQUIREMENTS} .
#--no-cache-dir prevent the downloads/ (small optimisation to lighten the image)
RUN python -m pip install --upgrade pip \
&& python -m pip install --no-cache-dir -r ${REQUIREMENTS} 

COPY . . 

EXPOSE 5000
#every 30s WGET if takes more than 5s TIME OUT, wait 3s when starting before tester and after 3 tries, DOCKER considers this image unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=3s --retries=3s \
    CMD wget -q0- http://localhost:5000/ || exit 1

CMD ["python","main.py"]