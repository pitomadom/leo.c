cmake_minimum_required(VERSION 3.22.1)
project("leo_jni")

set(LEO_ROOT ${CMAKE_SOURCE_DIR}/../../../../..)

add_library(leo_jni SHARED
    leo_jni.c
    ${LEO_ROOT}/leo.c
)

target_compile_options(leo_jni PRIVATE
    -O3 -ffast-math -DANDROID -DLEO_JNI
)

target_link_libraries(leo_jni
    log
    m
)

target_include_directories(leo_jni PRIVATE
    ${LEO_ROOT}
)
