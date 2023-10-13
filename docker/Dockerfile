# nvcomp Dockerfile
# (c) Stephen Bates, Eideticom Inc, 2022
#
# A Dockerfile that builds the nvcomp library and testbench apps so
# that they can be run inside a system running Docker with
# nvidia-docker2 support [1].
#
# Note that we copy the nvcomp benchmarks into /usr/local/bin so you
# can run them with something like this from outside the container.
#
# $ docker run --gpus all nvcomp:latest benchmark_lz4_synth
#
# [1]. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-docker

FROM nvidia/cuda:11.0-devel-ubuntu20.04
MAINTAINER <Stephen Bates>sbates@raithlin.com

ARG DEBIAN_FRONTEND=noninteractive

  # Install prerequistes

RUN apt-get update && apt-get -y install \
  git \
  mdm \
  wget \
  zlib1g-dev
  
  # Need to update cmake (sigh)

ARG CMAKE_VERSION=3.22.2

RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh

ENV PATH="/usr/bin/cmake/bin:${PATH}"

  # Install nvCOMP extensions

WORKDIR /opt/
RUN mkdir /opt/nvcomp-exts && cd /opt/nvcomp-exts \
      && wget http://developer.download.nvidia.com/compute/nvcomp/2.1/local_installers/nvcomp_exts_x86_64_ubuntu20.04-2.1.tar.gz \
      && tar xvfz nvcomp_exts_x86_64_ubuntu20.04-2.1.tar.gz

  # Clone, build and install nvcomp

ARG UBUNTU_VERSION=ubuntu20.04
ARG NVCOMP_EXT_CUDA_VERSION=11.4

WORKDIR /opt/
RUN git clone https://github.com/NVIDIA/nvcomp.git
WORKDIR /opt/nvcomp
RUN mkdir build
WORKDIR /opt/nvcomp/build
RUN cmake -DBUILD_BENCHMARKS=ON -DNVCOMP_EXTS_ROOT=/opt/nvcomp-exts/${UBUNTU_VERSION}/${NVCOMP_EXT_CUDA_VERSION} .. \
      && make -j $(ncpus) \
      && make install

  # Copy the benchmark files into /usr/local/bin so they easily
  # be run.

WORKDIR /opt/nvcomp/build/bin
RUN cp * /usr/local/bin