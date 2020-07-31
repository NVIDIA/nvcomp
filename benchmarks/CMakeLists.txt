file(GLOB EXAMPLE_SOURCES *.cpp)
include_directories(${CUDA_TOOLKIT_INCLUDE})

find_package(Threads REQUIRED)

include_directories(../src/cub)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
  get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  target_link_libraries(${BARE_NAME} nvcomp)
  if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(${BARE_NAME} PUBLIC "-pthread")
  endif()
  if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${BARE_NAME} "${CMAKE_THREAD_LIBS_INIT}")
  endif()
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
