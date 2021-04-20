file(GLOB GDEFLATE_CPU_SOURCES *gdeflate_cpu*.cpp *gdeflate_cpu*.cu)

# Add gdeflate CPU executables 
find_package(ZLIB)
# gdeflate CPU examples require gdeflate and zlib libraries
# skip gdeflate CPU examples if required libraries are missing
if (GDEFLATE_FOUND AND ZLIB_FOUND)
  foreach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
    # cut off suffixes
    string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
    string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
    get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
    add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp CUDA::cudart)
    if(THREADS_HAVE_PTHREAD_ARG)
      target_compile_options(${BARE_NAME} PUBLIC "-pthread")
    endif()
    if(CMAKE_THREAD_LIBS_INIT)
      target_link_libraries(${BARE_NAME} PRIVATE "${CMAKE_THREAD_LIBS_INIT}" CUDA::nvml)
    endif()
    target_include_directories(${BARE_NAME} PUBLIC ${GDEFLATE_INCLUDE_DIRS})
    target_link_libraries(${BARE_NAME} PUBLIC ${GDEFLATE_LIBRARIES})
    target_link_libraries(${BARE_NAME} PRIVATE ZLIB::ZLIB)
  endforeach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
else()
  message(WARNING "Cannot build gdeflate CPU example without gdeflate and zlib libraries.")
endif()