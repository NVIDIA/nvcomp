podTemplate (cloud:'sc-ipp-blossom-prod', yaml : """
apiVersion: v1
kind: Pod
metadata:
  labels:
    some-label: some-label-value
spec:
  containers:
  - name: cuda-devel
    image: nvcr.io/nvidia/cuda:11.1-devel-ubuntu18.04
    command:
    - cat
    resources:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
    restartPolicy: Never
    backoffLimit: 4
    tty: true
  nodeSelector:
    kubernetes.io/os: linux
    nvidia.com/gpu_type: A100_PCIE_40GB
""") {
  node(POD_LABEL) {
    container("cuda-devel") {
      stage("dependencies") {
        sh "apt-get update && apt-get install -qy git cmake && apt-get clean && rm -rf /var/lib/apt-lists/*"
      }
      stage("checkout") {
        checkout scm
        sh "git submodule update --init --recursive"
      }
      stage("build-debug") {
        gitlabCommitStatus(connection: gitLabConnection('nvidia-gitlab'), name: 'build-debug') {
          sh "mkdir build-debug && \
              cd build-debug/ && \
              cmake ../ -DDEVEL=1 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=1 && \
              make -j\$(grep -c ^processor /proc/cpuinfo)"
        }
      }
      stage("build-release") {
        gitlabCommitStatus(connection: gitLabConnection('nvidia-gitlab'), name: 'build-release') {
          sh "mkdir build-release && \
               cd build-release/ && \
               cmake ../ -DDEVEL=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=1 && \
               make -j\$(grep -c ^processor /proc/cpuinfo)"
        }
      }
      stage("unit-test-debug") {
        gitlabCommitStatus(connection: gitLabConnection('nvidia-gitlab'), name: 'unit-test-debug') {
          sh "cd build-debug && CTEST_OUTPUT_ON_FAILURE=1 make test"
        }
      }
      stage("unit-test-release") {
        gitlabCommitStatus(connection: gitLabConnection('nvidia-gitlab'), name: 'unit-test-release') {
          sh "cd build-release && CTEST_OUTPUT_ON_FAILURE=1 make test"
        }
      }
    }
  }
}
