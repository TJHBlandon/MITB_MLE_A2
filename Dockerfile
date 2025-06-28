# V2
FROM apache/airflow:2.11.0-python3.9

# ---------- OS packages ----------
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        curl \
        && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME directory and symlink the java binary there
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java || true

# Environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONPATH="/opt/airflow/scripts:${PYTHONPATH}"

# ---------- Python deps ----------
COPY requirements.txt /tmp/req.txt
USER airflow

# Install all requirements (including SMTP provider) from requirements.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

USER airflow

# ---------- Airflow home ----------
WORKDIR /opt/airflow