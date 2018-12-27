FROM python:3.6
RUN mkdir /code
WORKDIR /code
ADD text_classifier.py /code
ADD nn_text_classifier.py /code
ADD dt_text_classifier.py /code
ADD requirements.txt /code
RUN pip install -r requirements.txt


