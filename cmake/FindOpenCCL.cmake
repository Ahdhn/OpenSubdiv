# Try to find OpenCCL library and include path.
# Once done this will define
#
# OPENCCL_FOUND
# OPENCCL_INCLUDE_DIR
# OPENCCL_LIBRARIES
#

include(FindPackageHandleStandardArgs)

find_path(OPENCCL_INCLUDE_DIR
    NAMES
        OpenCCL.h
    PATHS
        /opt/local/include
        /home/mbdriscoll/opt/openccl/include
        ${OPENCCL_LOCATION}/include
        $ENV{OPENCCL_LOCATION}/include
    DOC
        "The directory where OpenCCL.h resides"
)

find_path( OPENCCL_LIBRARY_PATH
    NAMES
        libOpenCCL.a
    PATHS
        /home/mbdriscoll/opt/openccl/lib
        $ENV{OPENCCL_LOCATION}/lib
    DOC
        "The OpenCCL library"
)

set (OPENCCL_FOUND "NO")
if(OPENCCL_INCLUDE_DIR)
  if(OPENCCL_LIBRARY_PATH)
    set (OPENCCL_LIBRARIES "-L${OPENCCL_LIBRARY_PATH} -lOpenCCL -lmetis -lm")
    set (OPENCCL_INCLUDE_PATH ${OPENCCL_INCLUDE_DIR})
    set (OPENCCL_FOUND "YES")
  endif(OPENCCL_LIBRARY_PATH)
endif(OPENCCL_INCLUDE_DIR)

find_package_handle_standard_args(OPENCCL DEFAULT_MSG
    OPENCCL_INCLUDE_DIR
    OPENCCL_LIBRARIES
)
