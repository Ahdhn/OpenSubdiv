# Try to find OSKI library and include path.
# Once done this will define
#
# OSKI_FOUND
# OSKI_INCLUDE_DIR
# OSKI_LIBRARIES
#

find_path( OSKI_INCLUDE_DIR oski/oski.h
    ${OSKI_LOCATION}/include
    $ENV{OSKI_LOCATION}/include
    DOC "The directory where oski.h resides")

find_library( OSKI_oski_LIBRARY oski
    PATHS
    ${OSKI_LOCATION}/lib/oski
    $ENV{OSKI_LOCATION}/lib/oski
    DOC "The OSKI library")

set( OSKI_FOUND "NO" )

if(OSKI_INCLUDE_DIR)
  if(OSKI_oski_LIBRARY)
    set( OSKI_LIBRARIES
      ${OSKI_oski_LIBRARY}
    )
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

