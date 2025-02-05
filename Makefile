# Compiler and Flags
CXX = g++
CXXFLAGS = -Wall -ggdb3 -O5
LDFLAGS = -L. -lm

# Default target to create shared library
all: libenergy.so

# Compile shared library from energy.cpp and energy.hpp
libenergy.so: energy.cpp energy.hpp
	$(CXX) $(CXXFLAGS) -shared -o libenergy.so -fPIC energy.cpp

# Clean up the build artifacts
clean: FORCE
	@-rm libenergy.so

FORCE:
