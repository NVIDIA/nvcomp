file(GLOB EXAMPLE_SOURCES *.cpp *.cu)

set(GPU_ARCHS "60;70-real")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "9")
  set(GPU_ARCHS ${GPU_ARCHS} "75-real")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
  set(GPU_ARCHS ${GPU_ARCHS} "80")
endif()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  # cut off suffixes
  string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
  string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
  get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  set_property(TARGET ${BARE_NAME} PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  if (NOT MSVC)
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp CUDA::cudart CUDA::nvml)
  else()
    # Including CUDA::cudart on Windows resulted in multiple definition
    # linker errors for gdeflate and bitcomp benchmarks, since
    # cudart_static.lib is already included.
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp CUDA::nvml)
  endif()
  set_property(TARGET ${BARE_NAME} PROPERTY INSTALL_RPATH "\$ORIGIN/../lib")
  install(TARGETS ${BARE_NAME}
    RUNTIME DESTINATION bin)
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
