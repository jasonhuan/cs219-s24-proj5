CXX=g++
CXXFLAGS= -lgsl -lm -lgslcblas
CLASSES=

all: mp_solver

mp_solver: $(CLASSES)
	$(CXX) -o $@ $@.cpp $(CXXFLAGS)

clean:
	rm -rf *.o mp_solver

dist: tarball
tarball: clean
	tar -cvzf cs219-proj5.tar.gz mp_solver.cpp mp_solver.h Makefile Readme.md
