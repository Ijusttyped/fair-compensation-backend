FROM python:3.8

ENV ROOTDIR /app
WORKDIR $ROOTDIR

COPY ./requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ARG aws_key
ARG aws_secret

ENV PYTHONPATH "${ROOTDIR}:${ROOTDIR}/src"
ENV LOGGER "uvicorn"
ENV LABELS_PATH "artefacts/labels.json"
ENV MODEL_PATH "artefacts/model.joblib"
ENV PORT ${PORT:-8000}
ENV AWS_ACCESS_KEY_ID=$aws_key
ENV AWS_SECRET_ACCESS_KEY=$aws_secret

RUN dvc pull transform-features train

EXPOSE $PORT

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --log-level debug