CC=mpicc
NVCC=nvcc

OBJ_FLAGS=-O -c
LIBS=-lmpi

CUDA_INC=-I/usr/local/cuda/common/inc
CUDA_LIBS=-L/usr/local/cuda/lib64 -lcudart

OPENGL_LD=-DGL_GLEXT_PROTOTYPES -lGL -lglut -lGLU -lm


all: kmeans

kmeans: kmeans.o clusterassign.o visual.o
	$(CC) -o kmeans kmeans.o clusterassign.o visual.o $(CUDA_INC) $(CUDA_LIBS) $(LIBS) $(OPENGL_LD)

kmeans.o: kmeans.c
	$(CC) $(OBJ_FLAGS) kmeans.c

clusterassign.o: clusterassign.cu
	$(NVCC) $(OBJ_FLAGS) clusterassign.cu

visual.o:
	g++ $(OBJ_FLAGS) visual.cpp $(OPENGL_LD) 
clean:
	rm -rf kmeans *.o
