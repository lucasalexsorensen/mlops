FROM pytorch/torchserve:0.5.2-cpu
WORKDIR /home

RUN python3 -m pip install kornia==0.6.2

COPY outputs/model.pth model.pth
COPY src/models src/models/
COPY requirements/serve.txt requirements.txt

USER root
RUN torch-model-archiver \
    --model-name=mask \
    --version=1.0 \
    --model-file=src/models/model.py \
    --serialized-file=model.pth \
    --handler=src/models/model_handler.py \
    --extra-files=src/models/index_to_name.json\
    --export-path=. \
    --requirements-file=requirements.txt

EXPOSE 7080
EXPOSE 7081

CMD ["torchserve", \
    "--start", \
    "--ts-config=src/models/config.properties", \
    "--models", \
    "mask=mask.mar"]