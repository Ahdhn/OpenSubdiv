# Try to find MKL library and include path.
# Once done this will define
#
# MKL_FOUND
# MKL_INCLUDE_DIR
# MKL_LIBRARIES
#

include(FindPackageHandleStandardArgs)

find_path(MKL_INCLUDE_DIR
    NAMES
        mkl_spblas.h
        mkl.h
    PATHS
        /opt/intel/mkl/include
        /opt/intel/composerxe/mkl/include
        ${MKL_LOCATION}/include
        $ENV{MKL_LOCATION}/include
    DOC
        "The directory where mkl.h resides"
)

find_path( MKL_LIBRARY_PATH
    NAMES
        libmkl_core.a
    PATHS
        /opt/intel/mkl/lib
        /opt/intel/mkl/lib/intel64
        /opt/intel/composerxe/mkl/lib/intel64
        ${MKL_LOCATION}/lib
        $ENV{MKL_LOCATION}/lib
    DOC
        "The MKL core library"
)

set (MKL_FOUND "NO")
if(MKL_INCLUDE_DIR)
  if(MKL_LIBRARY_PATH)
    #set (MKL_LIBRARIES "-L${MKL_LIBRARY_PATH} -lmkl_intel -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm")
    set (MKL_LIBRARIES "-L${MKL_LIBRARY_PATH} -lmkl_intel -lmkl_intel_thread -lmkl_intel_ilp64 -lmkl_core -lpthread -lm")
    set (MKL_INCLUDE_PATH ${MKL_INCLUDE_DIR})
    set (MKL_FOUND "YES")
  endif(MKL_LIBRARY_PATH)
endif(MKL_INCLUDE_DIR)

find_package_handle_standard_args(MKL DEFAULT_MSG
    MKL_INCLUDE_DIR
    MKL_LIBRARIES
)
