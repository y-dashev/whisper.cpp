set(TARGET whisper-ptt)
add_executable(${TARGET} ptt.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
