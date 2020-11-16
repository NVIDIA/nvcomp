#!/user/bin/python
##
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# # Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# # Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# # Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import sys
import csv
import os
import subprocess

def allgather(filename, ftype, gpus, chunks, log):
        cmd = './bin/benchmark_allgather -f ' + str(filename) + ' -g ' + str(gpus) + ' -h ' + str(gpus*chunks) + ' -c none'
        cmd += ' -t ' + ftype
        log.write(cmd + "\n")
        result = subprocess.check_output(cmd, shell=True)
        log.write(result + "\n")
        nocomp = result.split()[-1]

        cmd = './bin/benchmark_allgather -f ' + str(filename) + ' -g ' + str(gpus) + ' -h ' + str(gpus*chunks) + ' -c lz4'
        cmd += ' -t ' + ftype
        log.write(cmd + "\n")
        result = subprocess.check_output(cmd, shell=True)
        log.write(result + "\n")
        lz4 = result.split()[-1]

        cmd = './bin/benchmark_allgather -f ' + str(filename) + ' -g ' + str(gpus) + ' -h ' + str(gpus*chunks) + ' -c cascaded'
        cmd += ' -t ' + ftype
        log.write(cmd + "\n")
        result = subprocess.check_output(cmd, shell=True)
        log.write(result + "\n")
        cascaded = result.split()[-1]
        return [filename, gpus, chunks, nocomp,lz4,cascaded]

def lz4(filename, ftype, log):
        cmd = './bin/benchmark_lz4 -f BIN:' + str(filename)
        log.write(cmd + "\n")
        result = subprocess.check_output(cmd, shell=True)
        log.write(result + "\n")
        ratio = result.split('compressed ratio: ')[1].split()[0]
        comp = result.split('compression throughput (GB/s): ')[1].split()[0]
        decomp = result.split('decompression throughput (GB/s): ')[1].split()[0]
        return [ratio, comp, decomp]

def cascaded(filename, ftype, log):
        cmd = './bin/benchmark_cascaded -f BIN:' + str(filename) + ' -t ' + ftype + ' -r 1 -d 1 -b 1'
        log.write(cmd + "\n")
        result = subprocess.check_output(cmd, shell=True)
        log.write(result + "\n")
        ratio = result.split('compressed ratio: ')[1].split()[0]
        comp = result.split('compression throughput (GB/s): ')[1].split()[0]
        decomp = result.split('decompression throughput (GB/s): ')[1].split()[0]
        return [ratio, comp, decomp]

if len(sys.argv) != 3:
	print "Usage: python sc20blog_runall.py <csv file with filenames,type pairs <type=int/long> <number of GPUs>"
	sys.exit(0)

filenames = sys.argv[1]
numgpus = sys.argv[2]

log = open('sc20blog-results.log', 'w')
with open('sc20blog-results.csv', 'w') as f:
        thewriter = csv.writer(f) 
        thewriter.writerow(['Filename', 'num GPUs', 'chunks per GPU', 
                            'allgather no-comp throughput', 'allgather LZ4 throughput', 'allgather Cascaded throughput',
                            'LZ4 comp ratio', 'LZ4 comp throughput', 'LZ4 decomp throughput',
                            'Cascaded comp ratio', 'Cascaded comp throughput', 'Cascaded decomp throughput'])

with open(filenames, 'r') as filenamesFile:
	reader = csv.reader(filenamesFile, delimiter=',')
	for row in reader:
		filename = row[0]
		ftype = row[1]
		for ngpus in range(2,int(numgpus)+1):
			print "Starting benchmark on file", filename, ", type", ftype, "- using", ngpus, "GPUs"
			with open('sc20blog-results.csv', 'a') as f:
        			thewriter = csv.writer(f) 
                		output_row = allgather(filename, ftype, ngpus, 1, log)
                        	output_row += lz4(filename, ftype, log)
                       		output_row += cascaded(filename, ftype, log)
				thewriter.writerow(output_row)
