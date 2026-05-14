# Build the fbx_modernize tool against AutoDesk FBX SDK 2020.3.9.
#
# Defaults assume the SDK was installed into ../../tmp/fbxsdk via the
# self-extracting installer (see docs/fbx3000_intel_handoff.md).
# Override FBXSDK_ROOT to point elsewhere.

FBXSDK_ROOT ?= $(realpath ../../tmp/fbxsdk)

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -I$(FBXSDK_ROOT)/include
LDFLAGS  ?= -L$(FBXSDK_ROOT)/lib/release
# Static-link libfbxsdk.a so the binary doesn't need LD_LIBRARY_PATH at run time.
# libxml2 + zlib are FBX SDK's external runtime deps; pthread/dl are pulled in
# by the static FBX archive itself.
LDLIBS   ?= $(FBXSDK_ROOT)/lib/release/libfbxsdk.a -lxml2 -lz -ldl -lpthread

fbx_modernize: fbx_modernize.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f fbx_modernize
