# Convenience Makefile: macOS / Linux (uses pkg-config sdl3) and Windows with a
# MinGW toolchain such as w64devkit. The fully portable build (incl.
# Windows/MSVC, and SDL3 fetched from source when it isn't installed) is
# CMakeLists.txt.
#
# Windows: SDL3_DIR must point at the x86_64-w64-mingw32 folder of the
# SDL3-devel-*-mingw package (override on the command line or in the env).

ifeq ($(OS),Windows_NT)
    CXX := g++
    SDL3_DIR ?= tools/SDL3-3.4.0/x86_64-w64-mingw32
    SDL3_CFLAGS := -I$(SDL3_DIR)/include
    SDL3_LIBS := -L$(SDL3_DIR)/lib -lSDL3 -mconsole
    PLATFORM_DEFS := -DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS
    PLATFORM_LIBS := -lcomdlg32 -lole32 -lshell32 -luuid
    EXE := .exe
else
    CXX := clang++
    SDL3_CFLAGS := $(shell pkg-config --cflags sdl3)
    SDL3_LIBS := $(shell pkg-config --libs sdl3)
    PLATFORM_DEFS :=
    PLATFORM_LIBS :=
    EXE :=
endif

CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter -MMD -MP $(SDL3_CFLAGS) $(PLATFORM_DEFS) \
            -Isrc -Ithird_party/imgui -Ithird_party/imgui/backends -Ithird_party/stb -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS -DIMGUI_ENABLE_TEST_ENGINE
LDFLAGS := $(SDL3_LIBS) $(PLATFORM_LIBS) -lpthread

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin
IMGUI := third_party/imgui

CORE_SRCS := $(filter-out $(SRC_DIR)/test_rdp.cpp, $(wildcard $(SRC_DIR)/*.cpp))
JIT_SRCS := $(filter-out $(SRC_DIR)/jit/jit_selftest_main.cpp, $(wildcard $(SRC_DIR)/jit/*.cpp))
CORE_SRCS += $(JIT_SRCS)
UI_SRCS := $(wildcard $(SRC_DIR)/ui/*.cpp)
# The GPU renderer (SDL_GPU compute shaders); the frontend and gpu_check use it.
GPU_SRCS := $(wildcard $(SRC_DIR)/gpu/*.cpp)
IMGUI_SRCS := $(IMGUI)/imgui.cpp $(IMGUI)/imgui_draw.cpp $(IMGUI)/imgui_tables.cpp $(IMGUI)/imgui_widgets.cpp \
              $(IMGUI)/backends/imgui_impl_sdl3.cpp $(IMGUI)/backends/imgui_impl_sdlrenderer3.cpp

GPU_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(GPU_SRCS))
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CORE_SRCS) $(UI_SRCS) $(GPU_SRCS)) \
        $(patsubst $(IMGUI)/%.cpp, $(BUILD_DIR)/imgui/%.o, $(IMGUI_SRCS))
# Headless RDP checker (tools/rdp_check.cpp): the core without main.cpp and the UI.
CHECK_OBJS := $(filter-out $(BUILD_DIR)/main.o, $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CORE_SRCS)))
DEPS := $(OBJS:.o=.d) $(BUILD_DIR)/test_rdp.d $(BUILD_DIR)/hires_exact.d \
        $(BUILD_DIR)/tools/rdp_check.d $(BUILD_DIR)/tools/rdp_check_exact.d $(BUILD_DIR)/tools/savestate_check.d $(BUILD_DIR)/tools/rsp_test.d
TARGET := $(BIN_DIR)/n64$(EXE)
TEST_TARGET := $(BIN_DIR)/test_rdp$(EXE)
CHECK_TARGET := $(BIN_DIR)/rdp_check$(EXE)
CHECK_EXACT_TARGET := $(BIN_DIR)/rdp_check_exact$(EXE)
SAVESTATE_CHECK_TARGET := $(BIN_DIR)/savestate_check$(EXE)
JIT_SELFTEST_TARGET := $(BIN_DIR)/jit_selftest$(EXE)
DEPS += $(BUILD_DIR)/jit/jit_selftest_main.d

.PHONY: all clean run test_rdp rdp_check rdp_check_exact savestate_check rsp_test jit_selftest gpu_check shaders
.DEFAULT_GOAL := all

# Interpreter-vs-JIT differential test (src/jit/jit_selftest_main.cpp).
jit_selftest: $(JIT_SELFTEST_TARGET)
	./$(JIT_SELFTEST_TARGET)

$(JIT_SELFTEST_TARGET): $(BUILD_DIR)/jit/jit_selftest_main.o $(CHECK_OBJS) | $(BIN_DIR)
	$(CXX) $^ -o $@ -lpthread

# GPU renderer vs CPU renderer, frame by frame (tools/gpu_check.cpp).
GPU_CHECK_TARGET := $(BIN_DIR)/gpu_check$(EXE)
gpu_check: $(GPU_CHECK_TARGET)
$(GPU_CHECK_TARGET): $(BUILD_DIR)/tools/gpu_check.o $(CHECK_OBJS) $(GPU_OBJS) | $(BIN_DIR)
	$(CXX) $^ -o $@ $(LDFLAGS)
$(BUILD_DIR)/tools/gpu_check.o: tools/gpu_check.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -w -c $< -o $@
DEPS += $(BUILD_DIR)/tools/gpu_check.d

# Regenerates src/gpu/shaders_gen.cpp after a shader change (needs glslangValidator and spirv-cross).
shaders:
	python3 tools/gen_shaders.py

# Headless stress test of the threaded core the frontend drives (tools/core_stress.cpp).
# Best built with checks, e.g.: make BUILD_DIR=build_chk BIN_DIR=bin_chk CXX="g++ -g -D_GLIBCXX_ASSERTIONS" core_stress
CORE_STRESS_TARGET := $(BIN_DIR)/core_stress$(EXE)
.PHONY: core_stress
core_stress: $(CORE_STRESS_TARGET)
$(CORE_STRESS_TARGET): $(BUILD_DIR)/tools/core_stress.o $(CHECK_OBJS) $(BUILD_DIR)/ui/emu_core.o $(BUILD_DIR)/ui/platform.o | $(BIN_DIR)
	$(CXX) $^ -o $@ $(LDFLAGS)
$(BUILD_DIR)/tools/core_stress.o: tools/core_stress.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Umain -c $< -o $@

# Host audio path checks: resampler quality and pacing (tools/audio_check.cpp).
AUDIO_CHECK_TARGET := $(BIN_DIR)/audio_check$(EXE)
.PHONY: audio_check
audio_check: $(AUDIO_CHECK_TARGET)
	./$(AUDIO_CHECK_TARGET)
$(AUDIO_CHECK_TARGET): $(BUILD_DIR)/tools/audio_check.o $(BUILD_DIR)/audio_stream.o | $(BIN_DIR)
	$(CXX) $^ -o $@
$(BUILD_DIR)/tools/audio_check.o: tools/audio_check.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Umain -c $< -o $@

# Console tools without SDL keep their own main().
$(BUILD_DIR)/jit/jit_selftest_main.o $(BUILD_DIR)/tools/rdp_check.o $(BUILD_DIR)/tools/rdp_check_exact.o \
$(BUILD_DIR)/tools/savestate_check.o $(BUILD_DIR)/tools/rsp_test.o: CXXFLAGS += -Umain

all: $(TARGET)

test_rdp: $(TEST_TARGET)

rdp_check: $(CHECK_TARGET)

rdp_check_exact: $(CHECK_EXACT_TARGET)

# Save states restore exactly the machine they were made from (tools/savestate_check.cpp).
savestate_check: $(SAVESTATE_CHECK_TARGET)

# Runs RSP memory images on the low-level RSP (tools/rsp_test.cpp).
rsp_test: $(BIN_DIR)/rsp_test$(EXE)

-include $(DEPS)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(TEST_TARGET): $(BUILD_DIR)/test_rdp.o $(BUILD_DIR)/rdp.o $(BUILD_DIR)/hires.o $(BUILD_DIR)/mi.o | $(BIN_DIR)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(CHECK_TARGET): $(BUILD_DIR)/tools/rdp_check.o $(CHECK_OBJS) | $(BIN_DIR)
	$(CXX) $^ -o $@ -lpthread

# The high-resolution pass stores exactly what RDRAM holds (HIRES_EXACT_TEST).
$(CHECK_EXACT_TARGET): $(BUILD_DIR)/tools/rdp_check_exact.o $(BUILD_DIR)/hires_exact.o $(filter-out $(BUILD_DIR)/hires.o, $(CHECK_OBJS)) | $(BIN_DIR)
	$(CXX) $^ -o $@ -lpthread

$(SAVESTATE_CHECK_TARGET): $(BUILD_DIR)/tools/savestate_check.o $(CHECK_OBJS) | $(BIN_DIR)
	$(CXX) $^ -o $@ -lpthread

$(BIN_DIR)/rsp_test$(EXE): $(BUILD_DIR)/tools/rsp_test.o $(CHECK_OBJS) | $(BIN_DIR)
	$(CXX) $^ -o $@ -lpthread

$(BUILD_DIR)/tools/rsp_test.o: tools/rsp_test.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/hires_exact.o: $(SRC_DIR)/hires.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -DHIRES_EXACT_TEST -c $< -o $@

$(BUILD_DIR)/tools/rdp_check.o: tools/rdp_check.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -w -c $< -o $@

$(BUILD_DIR)/tools/savestate_check.o: tools/savestate_check.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tools/rdp_check_exact.o: tools/rdp_check.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -w -DHIRES_EXACT_TEST -c $< -o $@

$(BUILD_DIR)/imgui/%.o: $(IMGUI)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -w -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

run: $(TARGET)
	./$(TARGET) $(ROM)
