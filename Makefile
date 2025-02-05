CXX = g++
CXXFLAGS = -Wall -ggdb3 -O5
LDFLAGS = -L. -lm

all: libenergy.so grad_w_armijo bfgs_w_classes bfgs_w_varargs

libenergy.so: energy.cpp energy.hpp
	$(CXX) $(CXXFLAGS) -shared -o libenergy.so -fPIC energy.cpp

grad_w_armijo: libenergy.so grad_w_armijo.o 
	$(CXX) $(CXXFLAGS) grad_w_armijo.o -o grad_w_armijo  $(LDFLAGS) -lenergy


bfgs_w_classes: bfgs_w_classes.o
	$(CXX) $(CXXFLAGS) bfgs_w_classes.o -o bfgs_w_classes  $(LDFLAGS)

bfgs_w_varargs: bfgs_w_varargs.o
	$(CC) $(CFFLAGS) bfgs_w_varargs.o -o bfgs_w_varargs  $(LDFLAGS)


clean: FORCE
	@-rm libenergy.so
	@-rm grad_w_armijo
	@-rm grad_w_varargs
	@-rm grad_w_classes

FORCE:
