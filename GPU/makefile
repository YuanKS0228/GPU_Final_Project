all: GPU CPU
	nvcc main.o gpu.o -code=sm_120 -arch=compute_120 -o main.exe
GPU:
	nvcc gpu.cu -c -code=sm_120 -arch=compute_120
CPU:
	nvcc main.cu -c
clean:
	rm *.o main.exe
