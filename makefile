
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__
main:
	nvcc -std=c++11 main.cu -Xcompiler -fopenmp -lgomp \
		-o bin/program\
	   	 $(LDFLAGS)

debug:
	nvcc -std=c++11 -g main.cu -Xcompiler -fopenmp -lgomp \
		-o bin/program\
	   	 $(LDFLAGS)
clean:
	rm test/bin/* bin/*
