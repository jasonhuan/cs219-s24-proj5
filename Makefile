CXX=g++
CXXFLAGS= -lgsl -lm -lgslcblas
CLASSES=

all: mp_solver

gpufit: mp_solver_gpufit

mp_solver: $(CLASSES)
	$(CXX) -o $@ $@.cpp $(CXXFLAGS)

mp_solver_gpufit:
	/usr/bin/c++ -O3 -DNDEBUG -std=gnu++14 mp_solver.cpp -o ./mp_solver_gpufit  -Wl,-rpath,/home/shared/gpufit-build/Gpufit /usr/local/lib/libGpufit.so /usr/lib/x86_64-linux-gnu/libcudart_static.a -ldl /usr/lib/x86_64-linux-gnu/librt.a -lgsl -lm -lgslcblas

clean:
	rm -rf *.o mp_solver mp_solver_gpufit

dist: tarball
tarball: clean
	tar -cvzf cs219-proj5.tar.gz mp_solver.cpp mp_solver.h Makefile Readme.md
