CC=mpicc
NVCC=nvcc

OBJ_FLAGS=-O -c
LIBS=-lmpi

CUDA_INC=-I/usr/local/cuda/common/inc
CUDA_LIBS=-L/usr/local/cuda/lib64 -lcudart

all: kmeans

kmeans: kmeans.o clusterassign.o
	$(CC) $(CUDA_INC) $(CUDA_LIBS) $(LIBS) kmeans.o clusterassign.o -o kmeans

kmeans.o: kmeans.c
	$(CC) $(OBJ_FLAGS) kmeans.c

clusterassign.o: clusterassign.cu
	$(NVCC) $(OBJ_FLAGS) clusterassign.cu

clean:
	rm -rf kmeans *.o
