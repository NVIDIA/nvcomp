#!/bin/bash

declare -A cascaded_algo_map
# "RLEs DELTAs USE_BP"
cascaded_algo_map["geometrycache.tar"]="1 0 1 int"
cascaded_algo_map["silesia.tar"]="0 0 1 char"
cascaded_algo_map["texturecache.tar"]="1 0 1 short"
cascaded_algo_map["mortgage-2009Q2-col0-long.bin"]="1 0 1 longlong"

declare -A lz4_datatype_map
lz4_datatype_map["geometrycache.tar"]="char"
lz4_datatype_map["silesia.tar"]="char"
lz4_datatype_map["texturecache.tar"]="char"
lz4_datatype_map["mortgage-2009Q2-col0-long.bin"]="int"

declare -A bitcomp_datatype_map
bitcomp_datatype_map["mortgage-2009Q2-col0-long.bin"]="uchar"
bitcomp_datatype_map["silesia.tar"]="ulonglong"
bitcomp_datatype_map["texturecache.tar"]="ushort"
bitcomp_datatype_map["geometrycache.tar"]="ulonglong"

declare -A display_fname_map
display_fname_map["geometrycache.tar"]="Graphics: Geometry Data"
display_fname_map["silesia.tar"]="Silesia"
display_fname_map["texturecache.tar"]="Graphics: Textures Data"
display_fname_map["mortgage-2009Q2-col0-long.bin"]="Data Analytics: INT Columns"

declare -a typed_algo_list=("lz4" "bitcomp")

function get_dtype()
{
  local fname="$1"
  local algo="$2"

  if [[ $algo == "bitcomp" ]]; then
    dtype="${bitcomp_datatype_map[$fname]}"
  elif [[ $algo == "lz4" ]]; then
    dtype="${lz4_datatype_map[$fname]}"
  else
    dtype="-1"
  fi
}

output_header() {
  echo ,,Compression Ratio,Compression Throughput,Decompression Throughput
  return 0
}

run_benchmark () {
  INPUT_FILE=$2
  CMD="$1 -f ${INPUT_FILE}"
  INPUT_FILENAME=`basename $INPUT_FILE`
  ALGO=$3
  VARIANT=$4
  DTYPE=$5
  BP="0"
  RLES="0"
  DELTAS="0"
  FILENAME="${LOGDIR}/$(basename $1)_$(basename $2).log"
  variant_str=""
  if [[ $VARIANT -gt "-1" ]]; then
    CMD="${CMD} -a ${VARIANT}"
    if [[ $ALGO == "bitcomp" ]]; then
      if [[ $VARIANT == "0" ]]; then
        variant_str="-default"
      else
        variant_str="-sparse"
      fi
    elif [[ $ALGO == "gdeflate" ]]; then
      if [[ $VARIANT == "0" ]]; then
        variant_str="-high-throughput"
      elif [[ $VARIANT == "1" ]]; then
        variant_str="-high-compression"
      else
        variant_str="-entropy-only"
      fi
    fi
  fi

  if [[ $ALGO == casc* ]]; then
    RLES="2"
    DELTAS="1"
    BP="1"
    # lookup into map of algorithms per 
    if [ cascaded_algo_map[$INPUT_FILENAME] ]; then
      declare -a vals=(${cascaded_algo_map[$INPUT_FILENAME]})
      RLES="${vals[0]}"
      DELTAS="${vals[1]}"
      BP="${vals[2]}"
      DTYPE="${vals[3]}"
    fi
    CMD="${CMD} -r ${RLES} -d ${DELTAS} -b ${BP}"
  fi
  
  if [[ $DTYPE -ne "-1" ]]; then
    CMD="${CMD} -t ${DTYPE}"
  fi

  # If the benchmark fails, rerun with a char  
  echo "The command is ${CMD}"
  timeout 120 ${CMD} > "${FILENAME}"
  # ${CMD} > "${FILENAME}"

  EXIT_STATUS=$?
  if [ $EXIT_STATUS -eq 124 ]; then
    echo "Process timed out"
    return 0
  fi
  
  bytes=$(awk '/^uncompressed /{print $3}' "${FILENAME}")
  ratio=$(awk '/compressed ratio:/{print $5}' "${FILENAME}")
  comp_throughput=$(awk '/^compression throughput /{print $4}' "${FILENAME}")
  decomp_throughput=$(awk '/^decompression throughput /{print $4}' "${FILENAME}")
  echo ,"${ALGO}${variant_str}",$ratio,$comp_throughput,$decomp_throughput >> $OUTPUT_FILE
  return 0
}

run_benchmarks () {
  BINARY=$1
  fname=$2
  algo=$3
  variant=$4
  dtype=$5
  if [[ $algo == "gdeflate" ]]; then
    for variant in {0,1,2}; do
      run_benchmark "$BINARY" $fname $algo $variant $dtype
    done
  elif [[ $algo == "bitcomp" ]]; then
    for variant in {0,1}; do
      run_benchmark "$BINARY" $fname $algo $variant $dtype
    done
  else 
    run_benchmark "$BINARY" $fname $algo $variant $dtype
  fi
}

## Start main function

if [ $# -lt 3 ]
  then
    echo "Usage:"
    echo "    $0 [directory] [output_file] [gpu]"
    exit 1
fi

DIR="$1"
OUTPUT_FILE="$2"
GPU="$3"

# Create a temp directory for all the logs
LOGDIR="$(mktemp -d)"
trap 'rm -rf -- "${LOGDIR}"' EXIT

# Run the benchmarks for all files in DIR
declare -a AlgoArray=("lz4" "snappy" "cascaded" "gdeflate" "bitcomp" "deflate" "ans" "zstd")
# declare -a AlgoArray=( "cascaded" "lz4" "gdeflate" )
# declare -a AlgoArray=("bitcomp" "gdeflate" "zstd")

#declare -a AlgoArray=("cascaded")
declare -a files=("mortgage-2009Q2-col0-long.bin" "silesia.tar" "texturecache.tar" "geometrycache.tar")
# declare -a files=("mortgage-2009Q2-float-columns.bin")
echo ,,$GPU > $OUTPUT_FILE
output_header >> $OUTPUT_FILE
for fname in "${files[@]}";
do
  echo "${display_fname_map[$fname]}" >> $OUTPUT_FILE
  base_filename=$fname
  fname="${DIR}/${fname}"
  for algo in "${AlgoArray[@]}"; 
    do
      BINARY="./bin/benchmark_${algo}_chunked"
      variant="-1"
      GROUP=""
      dtype=""

      get_dtype $base_filename $algo    
      
      run_benchmarks $BINARY $fname $algo $variant $dtype    
    done  
done
