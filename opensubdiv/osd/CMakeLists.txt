#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
#
#

find_package(IlmBase REQUIRED)
find_package(OpenGL)
find_package(GLEW)


include_directories(
    ${ILMBASE_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/opensubdiv
)

#-------------------------------------------------------------------------------
# source & headers
set(SOURCE_FILES
    cpuDispatcher.cpp
    cpuKernel.cpp
    kernelDispatcher.cpp
    mesh.cpp
)

set(KERNEL_FILES
)

set(INC_FILES
)

set(PUBLIC_HEADERS
    cpuDispatcher.h
    cpuKernel.h
    kernelDispatcher.h
    mesh.h
    vertex.h
)


#-------------------------------------------------------------------------------
# CPU (omp) code & dependencies
if(APPLE)
    set(PLATFORM_COMPILE_FLAGS
	-fopenmp
    )
    set(PLATFORM_LIBRARIES
	gomp
    )
elseif(UNIX)
    set(PLATFORM_COMPILE_FLAGS
        -fopenmp
        -fPIC
    )
    set(PLATFORM_LIBRARIES
        gomp
    )
elseif(WIN32 or WIN64)
    set(PLATFORM_COMPILE_FLAGS
        /openmp
    )       
endif()



#-------------------------------------------------------------------------------
# GL code & dependencies 
# note : (GLSL compute kernels require GL 4.2, which excludes APPLE) 
if( OPENGL_FOUND AND GLEW_FOUND AND (NOT APPLE) )
    list(APPEND SOURCE_FILES
        glslDispatcher.cpp
    )
    list(APPEND PUBLIC_HEADERS
        glslDispatcher.h
    )
    list(APPEND KERNEL_FILES
        glslKernel.glsl
    )
    list(APPEND PLATFORM_LIBRARIES
        ${OPENGL_LIBRARY}
        ${GLEW_LIBRARY}
    )	
endif()

# We want to use preprocessor include directives to include GLSL and OpenCL
# kernel source files in cpp files, but since the sources contain newline
# characters we would need raw string literals from C++11 to do this directly.
# To avoid depending on C++11 we instead use a small tool called "line_quote"
# to generate source files that are suitable for direct inclusion.
foreach(kernel_file ${KERNEL_FILES})

    string(REGEX REPLACE ".*[.](.*)" "\\1" extension ${kernel_file})

    if(NOT ${extension} STREQUAL "cu")

        string(REGEX REPLACE "(.*)[.].*" "\\1.inc" inc_file ${kernel_file})
        list(APPEND INC_FILES ${inc_file})
        
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
            COMMAND stringify ${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file}
                ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
            DEPENDS stringify ${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file}
        )
        
    endif()    
endforeach()

source_group("Kernels" FILES ${KERNEL_FILES})
source_group("Inc" FILES ${INC_FILES})

macro(_add_library target)
    if(CUDA_FOUND)
        cuda_add_library(${target} ${ARGN})
    else()
        add_library(${target} ${ARGN})
    endif()
endmacro()

_add_library(osd SHARED
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${KERNEL_FILES}
    ${INC_FILES}
)

target_link_libraries(osd
    ${OPENGL_LIBRARY}
    ${PLATFORM_LIBRARIES}
)