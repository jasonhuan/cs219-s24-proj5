CXX=g++
CXXFLAGS= -lgsl -lm -lgslcblas
CLASSES=

all: mp_solver

mp_solver: $(CLASSES) thread_pool.o
	$(CXX) -o $@ $@.cpp thread_pool.o $(CXXFLAGS)

thread_pool.o: thread_pool.cpp thread_pool.h
	$(CXX) -c thread_pool.cpp 

clean:
	rm -rf *.o mp_solver

dist: tarball
tarball: clean
	tar -cvzf cs219-proj5.tar.gz mp_solver.cpp mp_solver.h Makefile Readme.md
