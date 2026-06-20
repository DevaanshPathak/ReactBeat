FROM python:3.11-slim-bookworm AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libasound2 \
        libportaudio2 \
        libsndfile1 \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY packaging ./packaging
COPY reactbeat.spec ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]" \
    && pyinstaller --clean --noconfirm reactbeat.spec

FROM debian:bookworm-slim AS validate

COPY --from=build /app/dist/reactbeat /usr/local/bin/reactbeat

RUN /usr/local/bin/reactbeat --smoke-test --mode particles --width 18 --height 6 >/tmp/particles.txt
RUN /usr/local/bin/reactbeat --smoke-test --mode fluid --style aurora --width 18 --height 6 >/tmp/fluid.txt
RUN /usr/local/bin/reactbeat --diagnostics >/tmp/diagnostics.txt

FROM scratch AS artifact

COPY --from=validate /usr/local/bin/reactbeat /reactbeat
