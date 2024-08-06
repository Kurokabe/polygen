FROM puzlcloud/fastai:2.0.8-cuda10.1-cudnn7-jupyter-python3.7

COPY requirements.txt /tmp/requirements.txt
RUN sudo pip uninstall typing -y
RUN pip install -r /tmp/requirements.txt
# RUN pip install tensorflow==1.14 dm-sonnet==1.32 tensor2tensor==1.15.7

WORKDIR /tmp

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && tar -xf google-cloud-cli-linux-x86_64.tar.gz && ./google-cloud-sdk/install.sh -q

ENV PATH="/tmp/google-cloud-sdk/bin:${PATH}"

RUN mkdir /tmp/vertex_model
RUN mkdir /tmp/face_model

RUN gsutil cp gs://deepmind-research-polygen/vertex_model.tar.gz /tmp/vertex_model/
RUN gsutil cp gs://deepmind-research-polygen/face_model.tar.gz /tmp/face_model/
RUN tar xvfz /tmp/vertex_model/vertex_model.tar.gz -C /tmp/vertex_model/
RUN tar xvfz /tmp/face_model/face_model.tar.gz -C /tmp/face_model/

WORKDIR /workspaces