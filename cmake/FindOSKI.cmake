# Try to find OSKI library and include path.
# Once done this will define
#
# OSKI_FOUND
# OSKI_INCLUDE_DIR
# OSKI_LIBRARIES
#

# We're going to want to read the shared variables from site-modules-shared.txt
# and append them to the link line. This solution was suggested on:
# http://www.cmake.org/pipermail/cmake/2003-March/003486.html
SET(cat_prog cat)
IF(WIN32)
  IF(NOT UNIX)
    SET(cat_prog type)
  ENDIF(NOT UNIX)
ENDIF(WIN32)

find_path( OSKI_INCLUDE_DIR
    NAMES
        oski/oski.h
    PATHS
        ${OSKI_LOCATION}/include
        $ENV{OSKI_LOCATION}/include
    DOC
        "The directory where oski.h resides"
)

find_library( OSKI_oski_LIBRARY
    NAMES
        oski
    PATHS
        ${OSKI_LOCATION}/lib/oski
        $ENV{OSKI_LOCATION}/lib/oski
    DOC
        "The OSKI library"
)

if(OSKI_INCLUDE_DIR)
  if(OSKI_oski_LIBRARY)
    EXEC_PROGRAM(${cat_prog}
        ARGS ${OSKI_LOCATION}/lib/oski/site-modules-shared.txt
        OUTPUT_VARIABLE OSKI_LIBS_NEWLINES
    )
    STRING(REGEX REPLACE "\n" " " OSKI_LIBRARIES "-L${OSKI_LOCATION}/lib/oski ${OSKI_LIBS_NEWLINES}" )
    unset(OSKI_LIBS_NEWLINES) # causes parsing problems with CMakeCache.txt otherwise

    set( OSKI_FOUND "YES" )

    set (OSKI_LIBRARY ${OSKI_LIBRARIES})
    set (OSKI_INCLUDE_PATH ${OSKI_INCLUDE_DIR})

  endif(OSKI_oski_LIBRARY)
endif(OSKI_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OSKI DEFAULT_MSG
    OSKI_INCLUDE_DIR
    OSKI_LIBRARIES
)

mark_as_advanced(
  OSKI_INCLUDE_DIR
)
