# Try to find pOSKI library and include path.
# Once done this will define
#
# POSKI_FOUND
# POSKI_INCLUDE_DIR
# POSKI_LIBRARIES
#

find_path( POSKI_INCLUDE_DIR poski/poski.h
    ${POSKI_LOCATION}/include
    $ENV{POSKI_LOCATION}/include
    DOC "The directory where poski.h resides")

find_library( POSKI_poski_LIBRARY poski
    PATHS
    ${POSKI_LOCATION}/lib
    $ENV{POSKI_LOCATION}/lib
    DOC "The pOSKI library")

set( POSKI_FOUND "NO" )

if(POSKI_INCLUDE_DIR)
  if(POSKI_poski_LIBRARY)
    set( POSKI_LIBRARIES
      ${POSKI_poski_LIBRARY}
    )
    set( POSKI_FOUND "YES" )

    set (POSKI_LIBRARY ${POSKI_LIBRARIES})
    set (POSKI_INCLUDE_PATH ${POSKI_INCLUDE_DIR})

  endif(POSKI_poski_LIBRARY)
endif(POSKI_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(POSKI DEFAULT_MSG
    POSKI_INCLUDE_DIR
    POSKI_LIBRARIES
)

mark_as_advanced(
  POSKI_INCLUDE_DIR
)

