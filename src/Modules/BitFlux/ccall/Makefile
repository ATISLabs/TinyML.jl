CC = gcc
SOURCES = BitDense.c
LIBNAME = binarynet
CFLAGS = -Wall -fpic -shared -O3 -march=native

build: 
	$(CC) -o $(LIBNAME).so $(SOURCES) $(CFLAGS)

clean:
	-rm *.o
	-rm *.dylib
	-rm *.so
