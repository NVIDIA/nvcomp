#!/bin/bash

declare -A cascaded_algo_map
# "RLEs DELTAs USE_BP"
cascaded_algo_map["geometrycache.tar"]="1 0 1 int"
cascaded_algo_map["silesia.tar"]="0 0 1 char"
cascaded_algo_map["texturecache.tar"]="1 0 1 short"
cascaded_algo_map["mortgage-2009Q2-float-columns.bin"]="2 2 1 int"
cascaded_algo_map["mortgage-2009Q2-string-columns.bin"]="2 0 1 char"
cascaded_algo_map["mortgage-2009Q2-col0-long.bin"]="1 0 1 longlong"

declare -A datatype_map
datatype_map["geometrycache.tar"]="int"
datatype_map["silesia.tar"]="char"
datatype_map["texturecache.tar"]="char"
datatype_map["mortgage-2009Q2-float-columns.bin"]="int"
datatype_map["mortgage-2009Q2-string-columns.bin"]="char"
datatype_map["mortgage-2009Q2-col0-long.bin"]="int"

declare -a typed_algo_list=("lz4")

function get_group()
{
  local fname="$1"
  if [[ "$fname" == *"mortgage"* ]]; then
    GROUP="tabular"
  elif [[ "$fname" == *"silesia"* ]]; then
    GROUP="silesia"
  else 
    GROUP="graphics"
  fi  
}

function get_dtype()
{
  local fname="$1"
  local algo="$2"

  found_algo=0
  for typed_algo in $typed_algo_list; do
    if [[ $algo == $typed_algo ]]; then
      found_algo=1
      break
    fi
  done

  if [[ $found_algo == "1" ]]; then
    dtype="${datatype_map[$fname]}"
  else
    dtype="-1"
  fi
}

output_header() {
  echo dataset,uncompressed_bytes,compression_ratio,compression_throughput,decompression_throughput,algorithm,interface,algorithm_variant,gpu,rles,deltas,use_bp,dataset_group
  return 0
}

run_benchmark () {
  INPUT_FILE=$2
  CMD="$1 -f ${INPUT_FILE}"
  INPUT_FILENAME=`basename $INPUT_FILE`
  ALGO=$3
  IFC=$4
  VARIANT=$5
  DTYPE=$6
  GROUP=$7
  BP="0"
  RLES="0"
  DELTAS="0"
  FILENAME="${LOGDIR}/$(basename $1)_$(basename $2).log"

  if [[ $VARIANT -gt "-1" ]]; then
    CMD="${CMD} -a ${VARIANT}"
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
  echo $(basename $2),$bytes,$ratio,$comp_throughput,$decomp_throughput,$ALGO,$IFC,$VARIANT,$GPU,$RLES,$DELTAS,$BP,$GROUP >> $OUTPUT_FILE
  return 0
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
output_header > $OUTPUT_FILE
declare -a AlgoArray=("lz4" "snappy" "cascaded" "gdeflate" "bitcomp" "deflate" "ans")
# declare -a AlgoArray=( "cascaded" "lz4" "gdeflate" )
# declare -a AlgoArray=("lz4")

#declare -a AlgoArray=("cascaded")
for algo in "${AlgoArray[@]}"; 
do
  BINARY="./bin/benchmark_${algo}_chunked"
  for fname in ${DIR}/*
  do
    variant="-1"
    GROUP=""
    dtype=""

    base_filename=`basename $fname`
    get_group $base_filename
    get_dtype $base_filename $algo

    if [[ $algo == "gdeflate" ]]; then
      for variant in {0,1,2}; do
        run_benchmark $BINARY $fname $algo "LL" $variant $dtype $GROUP
      done
    else 
      run_benchmark $BINARY $fname $algo "LL" $variant $dtype $GROUP
    fi
  done

  BINARY="./bin/benchmark_hlif ${algo}"
  for fname in ${DIR}/*
  do
    if [[ $algo == "deflate" ]]; then
      continue
    fi

    variant="-1"
    GROUP=""
    dtype=""

    base_filename=`basename $fname`
    get_group $base_filename
    get_dtype $base_filename $algo    
    run_benchmark "$BINARY" $fname $algo "HL" $variant $dtype $GROUP
  done
done
