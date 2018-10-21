main:
	nvcc -std=c++11 main.cu -Xcompiler -fopenmp -lgomp\
	   	-o bin/program

clean:
	rm test/bin/* bin/*
