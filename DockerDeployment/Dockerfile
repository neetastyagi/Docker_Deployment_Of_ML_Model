FROM continuumio/anaconda3
MAINTAINER UNP, https://unp.education
COPY ./api_demo /usr/local/python/
EXPOSE 1313
WORKDIR /usr/local/python/
RUN pip install --upgrade pip
RUN pip install --upgrade Cython
RUN pip install -r requirements.txt
CMD python --version
CMD python predict_api.py

