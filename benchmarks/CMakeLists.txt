file(GLOB EXAMPLE_SOURCES *.cpp *.cu)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  # cut off suffixes
  string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
  string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
  get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  target_link_libraries(${BARE_NAME} PRIVATE nvcomp CUDA::cudart CUDA::nvml)
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
