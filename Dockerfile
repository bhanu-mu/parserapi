

FROM python:3.7-slim
WORKDIR $APP_HOME
COPY . .
RUN pip install flask nltk spacy==2.3.5 https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz pyresparser
EXPOSE 3004
ENV APP_HOME /app
CMD ["python", "main.py"]
