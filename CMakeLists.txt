cmake_minimum_required(VERSION 3.16)
project(orbit64 VERSION 1.0.0 LANGUAGES CXX)

# The version the app shows (About window): major.minor from project(VERSION), then the number of commits, which is the
# number the release workflow gives its releases (v<major>.<minor>.<commits on main>), and "-<branch>" for any branch
# other than main. The number comes from -DORBIT64_BUILD_NUMBER=, else from a .build_info file (the CI writes the commit
# count and the branch name into it before it builds, also for the builds inside VMs), else from git when the clone is
# complete; with none of them it is 0.
set(ORBIT64_BUILD_NUMBER "" CACHE STRING "Commit count in the version string (default: .build_info, else git)")
set(ORBIT64_BRANCH "" CACHE STRING "Branch name in the version string (default: .build_info, else git)")
if(ORBIT64_BUILD_NUMBER STREQUAL "" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.build_info")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/.build_info" _orbit64_info)
    list(LENGTH _orbit64_info _orbit64_info_len)
    if(_orbit64_info_len GREATER 0)
        list(GET _orbit64_info 0 ORBIT64_BUILD_NUMBER)
    endif()
    if(_orbit64_info_len GREATER 1)
        list(GET _orbit64_info 1 ORBIT64_BRANCH)
    endif()
endif()
if(ORBIT64_BUILD_NUMBER STREQUAL "")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --is-shallow-repository
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE _orbit64_shallow
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_orbit64_shallow STREQUAL "false")
            execute_process(COMMAND "${GIT_EXECUTABLE}" rev-list --count HEAD
                            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE ORBIT64_BUILD_NUMBER
                            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
                            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE ORBIT64_BRANCH
                            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        endif()
    endif()
endif()
if(ORBIT64_BUILD_NUMBER STREQUAL "")
    set(ORBIT64_BUILD_NUMBER 0)
endif()
set(ORBIT64_VERSION_STRING "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${ORBIT64_BUILD_NUMBER}")
if(NOT ORBIT64_BRANCH STREQUAL "" AND NOT ORBIT64_BRANCH STREQUAL "main" AND NOT ORBIT64_BRANCH STREQUAL "HEAD")
    string(REGEX REPLACE "[^A-Za-z0-9._-]" "-" _orbit64_branch_safe "${ORBIT64_BRANCH}")
    set(ORBIT64_VERSION_STRING "${ORBIT64_VERSION_STRING}-${_orbit64_branch_safe}")
endif()
message(STATUS "Orbit64 version ${ORBIT64_VERSION_STRING}")

# Orbit64 - Nintendo 64 emulator with a cross-platform (Windows / macOS / Linux)
# desktop frontend built on SDL3 + Dear ImGui. Graphics go through SDL_GPU
# (Metal on macOS, Vulkan or Direct3D 12 on Windows, Vulkan on Linux), which
# also runs the high-resolution RDP renderer's compute shaders.
#
#   cmake -S . -B out -DCMAKE_BUILD_TYPE=Release
#   cmake --build out --config Release
#
# SDL3 (3.4 or newer) is the only external dependency. An installed copy is
# used when CMake finds one:
#   Windows : vcpkg install sdl3   (then -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake)
#   macOS   : brew install sdl3
#   Linux   : apt install libsdl3-dev / dnf install SDL3-devel / pacman -S sdl3
# Otherwise (or with -DORBIT64_VENDOR_SDL3=ON) SDL3 is downloaded and built
# from source as part of the project.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ORBIT64_WIN_CONSOLE "Windows: keep a console window (useful for --headless and logs)" OFF)
option(ORBIT64_VENDOR_SDL3 "Always build SDL3 from source instead of using an installed copy" OFF)
option(ORBIT64_SDL3_STATIC "Link SDL3 statically when it is built from source (self-contained release binaries)" OFF)
set(ORBIT64_SDL3_VERSION "3.4.16" CACHE STRING "SDL3 release built when no installed SDL3 is used")

if(NOT ORBIT64_VENDOR_SDL3)
    find_package(SDL3 3.4 CONFIG QUIET)
endif()
if(NOT SDL3_FOUND)
    message(STATUS "Orbit64: building SDL3 ${ORBIT64_SDL3_VERSION} from source")
    include(FetchContent)
    if(ORBIT64_SDL3_STATIC)
        set(SDL_SHARED OFF CACHE BOOL "" FORCE)
        set(SDL_STATIC ON CACHE BOOL "" FORCE)
    else()
        set(SDL_SHARED ON CACHE BOOL "" FORCE)
        set(SDL_STATIC OFF CACHE BOOL "" FORCE)
    endif()
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    # The default release is checked against a hash; another version has none to compare with.
    set(ORBIT64_SDL3_HASH "")
    if(ORBIT64_SDL3_VERSION STREQUAL "3.4.16")
        set(ORBIT64_SDL3_HASH "URL_HASH;SHA256=7322236cd12090c3eb40b9728be4d49c76f66ad17d04369584d4ecad5cf77c68")
    endif()
    FetchContent_Declare(SDL3
        URL "https://github.com/libsdl-org/SDL/releases/download/release-${ORBIT64_SDL3_VERSION}/SDL3-${ORBIT64_SDL3_VERSION}.tar.gz"
        ${ORBIT64_SDL3_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(SDL3)
endif()
find_package(Threads REQUIRED)

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/test_rdp.cpp")
file(GLOB JIT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/*.cpp")
list(REMOVE_ITEM JIT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/jit/jit_selftest_main.cpp")
list(APPEND CORE_SOURCES ${JIT_SOURCES})
file(GLOB UI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ui/*.cpp")
# The GPU renderer (SDL_GPU compute shaders, prebuilt into src/gpu/shaders_gen.cpp
# by tools/gen_shaders.py).
file(GLOB GPU_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cpp")

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer3.cpp)

if(ANDROID)
    add_library(n64 SHARED ${CORE_SOURCES} ${UI_SOURCES} ${GPU_SOURCES} ${IMGUI_SOURCES})
else()
    add_executable(n64 ${CORE_SOURCES} ${UI_SOURCES} ${GPU_SOURCES} ${IMGUI_SOURCES})
endif()
set_target_properties(n64 PROPERTIES OUTPUT_NAME "orbit64")
target_include_directories(n64 PRIVATE src ${IMGUI_DIR} ${IMGUI_DIR}/backends third_party/stb)
# IMGUI_ENABLE_TEST_ENGINE compiles in Dear ImGui's item hooks used by the UI
# clickability audit (src/ui/ui_audit.cpp); they are inactive unless the audit runs.
target_compile_definitions(n64 PRIVATE IMGUI_DISABLE_OBSOLETE_FUNCTIONS IMGUI_ENABLE_TEST_ENGINE ORBIT64_VERSION="${ORBIT64_VERSION_STRING}")

target_link_libraries(n64 PRIVATE SDL3::SDL3)
if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.21 AND NOT (ORBIT64_SDL3_STATIC AND NOT SDL3_FOUND))
    # SDL3.dll next to orbit64.exe, wherever it came from.
    add_custom_command(TARGET n64 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:n64> $<TARGET_FILE_DIR:n64>
        COMMAND_EXPAND_LISTS)
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
# (--gpu-exact runs the bit-exact RDP on the GPU.)
add_executable(savestate_check EXCLUDE_FROM_ALL tools/savestate_check.cpp ${CHECK_SOURCES} ${GPU_SOURCES})
target_include_directories(savestate_check PRIVATE src third_party/stb)
target_link_libraries(savestate_check PRIVATE SDL3::SDL3 Threads::Threads)

# Plays a ROM headless and records how it reacts to input (tools/game_probe.cpp,
# driven by tools/game_status.py), not built by default.
add_executable(game_probe EXCLUDE_FROM_ALL tools/game_probe.cpp ${CHECK_SOURCES})
target_include_directories(game_probe PRIVATE src third_party/stb)
target_link_libraries(game_probe PRIVATE Threads::Threads)

# Plays recorded routes and judges what is drawn (tools/game_qa.cpp, driven by
# tools/qa.py), not built by default.
add_executable(game_qa EXCLUDE_FROM_ALL tools/game_qa.cpp ${CHECK_SOURCES})
target_include_directories(game_qa PRIVATE src third_party/stb)
target_link_libraries(game_qa PRIVATE Threads::Threads)

# Interpreter-vs-JIT differential test (src/jit/jit_selftest_main.cpp), not built by default.
add_executable(jit_selftest EXCLUDE_FROM_ALL src/jit/jit_selftest_main.cpp ${CHECK_SOURCES})
target_include_directories(jit_selftest PRIVATE src third_party/stb)
target_link_libraries(jit_selftest PRIVATE Threads::Threads)

# GPU renderer vs CPU renderer, frame by frame (tools/gpu_check.cpp), not built by default.
add_executable(gpu_check EXCLUDE_FROM_ALL tools/gpu_check.cpp ${CHECK_SOURCES} ${GPU_SOURCES})
target_include_directories(gpu_check PRIVATE src third_party/stb)
target_link_libraries(gpu_check PRIVATE SDL3::SDL3 Threads::Threads)
if(WIN32)
    target_compile_definitions(gpu_check PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# Bit-exact RDP on the GPU vs on the CPU (tools/exact_gpu_check.cpp), not built by default.
add_executable(exact_gpu_check EXCLUDE_FROM_ALL tools/exact_gpu_check.cpp ${CHECK_SOURCES} ${GPU_SOURCES})
target_include_directories(exact_gpu_check PRIVATE src third_party/stb)
target_link_libraries(exact_gpu_check PRIVATE SDL3::SDL3 Threads::Threads)
if(WIN32)
    target_compile_definitions(exact_gpu_check PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(savestate_check PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# Host audio path checks (tools/audio_check.cpp), not built by default.
add_executable(audio_check EXCLUDE_FROM_ALL tools/audio_check.cpp src/audio_stream.cpp)
target_include_directories(audio_check PRIVATE src)

# Optional RDP unit test (unchanged from the core project).
add_executable(test_rdp EXCLUDE_FROM_ALL src/test_rdp.cpp src/rdp.cpp src/hires.cpp src/mi.cpp)
target_include_directories(test_rdp PRIVATE src)
target_link_libraries(test_rdp PRIVATE Threads::Threads)
