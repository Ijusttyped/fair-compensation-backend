FROM python:3.8

ENV ROOTDIR /app
WORKDIR $ROOTDIR

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

ENV PYTHONPATH "${ROOTDIR}:${ROOTDIR}/src"
ENV LOGGER "uvicorn"
ENV LABELS_PATH "artefacts/labels.json"
ENV MODEL_PATH "artefacts/model.joblib"
ENV PORT ${PORT:-8000}

EXPOSE $PORT

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --log-level debug