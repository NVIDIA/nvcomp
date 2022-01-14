file(GLOB EXAMPLE_SOURCES *.cpp *.c)

function(add_test_file EXAMPLE_SOURCE)
  get_filename_component(BARE_NAME "${EXAMPLE_SOURCE}" NAME_WE)
  add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
  target_link_libraries(${BARE_NAME} PRIVATE nvcomp)
  add_test(NAME ${BARE_NAME} COMMAND ${BARE_NAME})
endfunction()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  get_filename_component(BARE_NAME "${EXAMPLE_NAME}" NAME_WE)
  add_test_file(${EXAMPLE_SOURCE} ${EXAMPLE_NAME})
endforeach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
