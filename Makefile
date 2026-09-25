# Convenience Makefile for macOS / Linux (uses sdl2-config).
# The portable build (incl. Windows/MSVC) is CMakeLists.txt.

CXX := clang++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter -MMD -MP $(shell sdl2-config --cflags) \
            -Isrc -Ithird_party/imgui -Ithird_party/imgui/backends -Ithird_party/stb -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS -DIMGUI_ENABLE_TEST_ENGINE
LDFLAGS := $(shell sdl2-config --libs) -lpthread

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin
IMGUI := third_party/imgui

CORE_SRCS := $(filter-out $(SRC_DIR)/test_rdp.cpp, $(wildcard $(SRC_DIR)/*.cpp))
JIT_SRCS := $(filter-out $(SRC_DIR)/jit/jit_selftest_main.cpp, $(wildcard $(SRC_DIR)/jit/*.cpp))
CORE_SRCS += $(JIT_SRCS)
UI_SRCS := $(wildcard $(SRC_DIR)/ui/*.cpp)
IMGUI_SRCS := $(IMGUI)/imgui.cpp $(IMGUI)/imgui_draw.cpp $(IMGUI)/imgui_tables.cpp $(IMGUI)/imgui_widgets.cpp \
              $(IMGUI)/backends/imgui_impl_sdl2.cpp $(IMGUI)/backends/imgui_impl_sdlrenderer2.cpp

OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CORE_SRCS) $(UI_SRCS)) \
        $(patsubst $(IMGUI)/%.cpp, $(BUILD_DIR)/imgui/%.o, $(IMGUI_SRCS))
# Headless RDP checker (tools/rdp_check.cpp): the core without main.cpp and the UI.
CHECK_OBJS := $(filter-out $(BUILD_DIR)/main.o, $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CORE_SRCS)))
DEPS := $(OBJS:.o=.d) $(BUILD_DIR)/test_rdp.d $(BUILD_DIR)/hires_exact.d \
        $(BUILD_DIR)/tools/rdp_check.d $(BUILD_DIR)/tools/rdp_check_exact.d
TARGET := $(BIN_DIR)/n64
TEST_TARGET := $(BIN_DIR)/test_rdp
CHECK_TARGET := $(BIN_DIR)/rdp_check
CHECK_EXACT_TARGET := $(BIN_DIR)/rdp_check_exact

.PHONY: all clean run test_rdp rdp_check rdp_check_exact

all: $(TARGET)

test_rdp: $(TEST_TARGET)

rdp_check: $(CHECK_TARGET)

rdp_check_exact: $(CHECK_EXACT_TARGET)

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

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/hires_exact.o: $(SRC_DIR)/hires.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -DHIRES_EXACT_TEST -c $< -o $@

$(BUILD_DIR)/tools/rdp_check.o: tools/rdp_check.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -w -c $< -o $@

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
