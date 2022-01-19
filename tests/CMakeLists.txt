file(GLOB EXAMPLE_SOURCES *.cpp *.c)

function(add_test_file EXAMPLE_SOURCE)
  get_filename_component(BARE_NAME "${EXAMPLE_SOURCE}" NAME_WE)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  if (NOT MSVC)
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp CUDA::cudart)
  else()
    # Including CUDA::cudart on Windows resulted in multiple definition
    # linker errors for gdeflate and bitcomp tests, since
    # cudart_static.lib is already included.
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp)
  endif()
  add_test(NAME ${BARE_NAME} COMMAND ${BARE_NAME})
endfunction()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  get_filename_component(BARE_NAME "${EXAMPLE_NAME}" NAME_WE)
  add_test_file(${EXAMPLE_SOURCE} ${EXAMPLE_NAME})
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
