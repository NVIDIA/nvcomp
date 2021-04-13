include( FindPackageHandleStandardArgs )

# Checks an environment variable; note that the first check
# does not require the usual CMake $-sign.
if( DEFINED ENV{GDEFLATE_ROOT} )
  set( GDEFLATE_ROOT "$ENV{GDEFLATE_ROOT}" )
ENDIF()

find_path(
  GDEFLATE_INCLUDE_DIR gdeflate.h
  HINTS ${GDEFLATE_ROOT}
  PATH_SUFFIXES include
)

find_library( GDEFLATE_LIBRARY
  NAMES gdeflate
  HINTS ${GDEFLATE_ROOT}
  PATH_SUFFIXES lib
)

find_package_handle_standard_args( gdeflate DEFAULT_MSG
  GDEFLATE_INCLUDE_DIR
  GDEFLATE_LIBRARY
)

if( GDEFLATE_FOUND )
  set( GDEFLATE_INCLUDE_DIRS ${GDEFLATE_INCLUDE_DIR} )
  set( GDEFLATE_LIBRARIES ${GDEFLATE_LIBRARY} )

  mark_as_advanced(
    GDEFLATE_LIBRARY
    GDEFLATE_INCLUDE_DIR
    GDEFLATE_ROOT
  )
else()
  set( GDEFLATE_ROOT "" CACHE STRING
    "An optional hint to a directory for finding `gdeflate`"
  )
endif()
