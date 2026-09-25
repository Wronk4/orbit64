cmake_minimum_required(VERSION 3.16)
project(orbit64 VERSION 1.0.0 LANGUAGES CXX)

# Orbit64 - Nintendo 64 emulator with a cross-platform (Windows / macOS / Linux)
# desktop frontend built on SDL2 + Dear ImGui.
#
#   cmake -S . -B out -DCMAKE_BUILD_TYPE=Release
#   cmake --build out --config Release
#
# SDL2 is the only external dependency:
#   Windows : vcpkg install sdl2   (then -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake)
#   macOS   : brew install sdl2
#   Linux   : apt install libsdl2-dev / dnf install SDL2-devel / pacman -S sdl2

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ORBIT64_WIN_CONSOLE "Windows: keep a console window (useful for --headless and logs)" OFF)

find_package(SDL2 CONFIG QUIET)
if(NOT SDL2_FOUND)
    # Fallback for systems that ship only a pkg-config file.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2_PC REQUIRED IMPORTED_TARGET sdl2>=2.0.18)
endif()
find_package(Threads REQUIRED)

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/test_rdp.cpp")
file(GLOB JIT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/*.cpp")
list(REMOVE_ITEM JIT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/jit_selftest_main.cpp")
list(APPEND CORE_SOURCES ${JIT_SOURCES})
file(GLOB UI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/*.cpp")

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer2.cpp)

add_executable(n64 ${CORE_SOURCES} ${UI_SOURCES} ${IMGUI_SOURCES})
set_target_properties(n64 PROPERTIES OUTPUT_NAME "orbit64")
target_include_directories(n64 PRIVATE src ${IMGUI_DIR} ${IMGUI_DIR}/backends third_party/stb)
# IMGUI_ENABLE_TEST_ENGINE compiles in Dear ImGui's item hooks used by the UI
# clickability audit (src/ui/ui_audit.cpp); they are inactive unless the audit runs.
target_compile_definitions(n64 PRIVATE IMGUI_DISABLE_OBSOLETE_FUNCTIONS IMGUI_ENABLE_TEST_ENGINE)

# SDL2 exposes imported targets in its CMake config; older distro packages only
# provide variables, so support both.
if(TARGET SDL2::SDL2main)
    target_link_libraries(n64 PRIVATE SDL2::SDL2main)
endif()
if(TARGET SDL2::SDL2)
    target_link_libraries(n64 PRIVATE SDL2::SDL2)
elseif(TARGET SDL2::SDL2-static)
    target_link_libraries(n64 PRIVATE SDL2::SDL2-static)
elseif(TARGET PkgConfig::SDL2_PC)
    target_link_libraries(n64 PRIVATE PkgConfig::SDL2_PC)
else()
    target_include_directories(n64 PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(n64 PRIVATE ${SDL2_LIBRARIES})
endif()
target_link_libraries(n64 PRIVATE Threads::Threads)

if(WIN32)
    # comdlg32/ole32/shell32: native file dialogs and "Show in Explorer".
    target_link_libraries(n64 PRIVATE comdlg32 ole32 shell32 uuid)
    target_compile_definitions(n64 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    if(NOT ORBIT64_WIN_CONSOLE)
        set_target_properties(n64 PROPERTIES WIN32_EXECUTABLE ON)
    endif()
endif()

if(MSVC)
    # UTF-8 source/execution charset (UI strings use UTF-8), large object for embedded fonts.
    target_compile_options(n64 PRIVATE /utf-8 /W3 /bigobj /permissive-)
else()
    target_compile_options(n64 PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Headless RDP checker (tools/rdp_check.cpp), not built by default. The
# _exact variant runs the high-resolution pass even at scale 1 and makes it
# store exactly what RDRAM holds, so its output must match the native renderer.
set(CHECK_SOURCES ${CORE_SOURCES})
list(REMOVE_ITEM CHECK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
foreach(check rdp_check rdp_check_exact)
    add_executable(${check} EXCLUDE_FROM_ALL tools/rdp_check.cpp ${CHECK_SOURCES})
    target_include_directories(${check} PRIVATE src third_party/stb)
    target_link_libraries(${check} PRIVATE Threads::Threads)
endforeach()
target_compile_definitions(rdp_check_exact PRIVATE HIRES_EXACT_TEST)
# Save state round trip (tools/savestate_check.cpp), not built by default.
add_executable(savestate_check EXCLUDE_FROM_ALL tools/savestate_check.cpp ${CHECK_SOURCES})
target_include_directories(savestate_check PRIVATE src third_party/stb)
target_link_libraries(savestate_check PRIVATE Threads::Threads)

# Host audio path checks (tools/audio_check.cpp), not built by default.
add_executable(audio_check EXCLUDE_FROM_ALL tools/audio_check.cpp src/audio_stream.cpp)
target_include_directories(audio_check PRIVATE src)

# Optional RDP unit test (unchanged from the core project).
add_executable(test_rdp EXCLUDE_FROM_ALL src/test_rdp.cpp src/rdp.cpp src/hires.cpp src/mi.cpp)
target_include_directories(test_rdp PRIVATE src)
target_link_libraries(test_rdp PRIVATE Threads::Threads)
if(TARGET SDL2::SDL2)
    target_link_libraries(test_rdp PRIVATE SDL2::SDL2)
endif()
