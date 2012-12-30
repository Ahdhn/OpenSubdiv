# *** headlessViewer ***

set(PLATFORM_LIBRARIES
    ${OSD_LINK_TARGET}
    ${GLEW_LIBRARY}
    ${GLUT_LIBRARIES}
)

include_directories(
    ${PROJECT_SOURCE_DIR}/opensubdiv
    ${PROJECT_SOURCE_DIR}/regression
    ${GLEW_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
)

if ( MKL_FOUND )
    list(APPEND PLATFORM_LIBRARIES ${MKL_LIBRARIES} )
    include_directories( ${MKL_INCLUDE_PATH} )
endif()

if ( CUDA_FOUND )
    list(APPEND PLATFORM_LIBRARIES
        -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcusparse
    )
endif()

#-------------------------------------------------------------------------------
# Shader Stringification
# We want to use preprocessor include directives to include GLSL and OpenCL
# shader source files in cpp files, but since the sources contain newline
# characters we would need raw string literals from C++11 to do this directly.
# To avoid depending on C++11 we instead use a small tool called "line_quote"
# to generate source files that are suitable for direct inclusion.
foreach(shader_file ${SHADER_FILES})

    string(REGEX REPLACE ".*[.](.*)" "\\1" extension ${shader_file})

    string(REGEX REPLACE "(.*)[.].*" "\\1.inc" inc_file ${shader_file})
    list(APPEND INC_FILES ${inc_file})

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
        COMMAND stringify ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}
            ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
        DEPENDS stringify ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}
    )
endforeach()

_add_glut_executable(headlessViewer
    viewer.cpp
    ${SHADER_FILES}
    ${INC_FILES}
)

target_link_libraries(headlessViewer
    ${PLATFORM_LIBRARIES}
)
