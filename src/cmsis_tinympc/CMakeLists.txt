# add_library (tinympc SHARED
add_library (cmsis_tinympc
    cmsis_test.cpp
    )

target_include_directories (cmsis_tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) # Include src/ directory instead of src/tinympc/