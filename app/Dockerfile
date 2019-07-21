FROM neox1993/fly-track

WORKDIR /code

ADD . /code

ENV DISPLAY=$DISPLAY

RUN pip install -r requirements.txt

CMD ["python", "./object_tracker.py"]