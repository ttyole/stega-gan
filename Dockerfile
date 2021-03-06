FROM tensorflow/tensorflow:latest-gpu

COPY ./requirements.txt /tf/requirements.txt

RUN pip install  --ignore-installed --default-timeout=100 -r /tf/requirements.txt
RUN apt-get -y install python-tk vim

COPY ./models/utility/stc /tf/app/models/utility/stc
RUN make -C /tf/app/models/utility/stc

COPY . /tf/app

WORKDIR /tf/app

EXPOSE 80