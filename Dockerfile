FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs wget libgtest-dev libboost-test-dev build-essential zlib1g-dev liblz4-dev&& \
    apt-get -y install  python3-dev python3-pip python3-setuptools &&\
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /home/
RUN python3 -m pip install -r /home/requirements.txt

RUN mkdir -p /usr/ext
WORKDIR /tmp/
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.4/cmake-3.21.4-linux-x86_64.sh -O cmake-install.sh
#COPY cmake-3.21.4-linux-x86_64.sh /tmp/cmake-install.sh
RUN   chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh
#
ENV PATH="/usr/bin/cmake/bin:${PATH}"
RUN cd /usr/ext && \
    wget http://developer.download.nvidia.com/compute/nvcomp/2.1/local_installers/nvcomp_exts_x86_64_ubuntu20.04-2.1.tar.gz &&\
    tar -xzf nvcomp_exts_x86_64_ubuntu20.04-2.1.tar.gz && rm nvcomp_exts_x86_64_ubuntu20.04-2.1.tar.gz
#
COPY . /usr/nvcomp/
WORKDIR /usr/nvcomp/
RUN mkdir -p /usr/nvcomp/build
WORKDIR /usr/nvcomp/build
RUN cd /usr/nvcomp/build && cmake -DBUILD_EXAMPLES=ON -DBUILD_BENCHMARKS=ON -DNVCOMP_EXTS_ROOT=/usr/ext/ubuntu20.04/11.4 .. && make -j && make install
#
##COPY tpch-dbgen /usr/tpch-dbgen
##RUN cd /usr/tpch-dbgen && make -j && ./dbgen -s 1
##RUN mkdir -p /usr/data && cd /usr/data &&\
##    wget http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000.tgz &&\
##    tar zxvf mortgage_2000.tgz \
#
##RUN python3 benchmarks/text_to_binary.py /usr/tpch-dbgen/lineitem.tbl 10 string column_data.bin '|'
