all: linux

remake: clean all

mac:
	g++ visual.cpp -DGL_GLEXT_PROTOTYPES -framework OpenGL -framework GLUT -o visual

linux:
	g++ visual.cpp -DGL_GLEXT_PROTOTYPES -lGL -lglut -lGLU -lm -o visual

clean:
	rm -f *~
	rm -f visual
	rm -f *.o

