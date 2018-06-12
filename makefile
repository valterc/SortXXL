# sortXXL makefile
# Valter Costa, João Sousa, João Órfão, ESTG IPleiria

CUDA_INSTALL_PATH ?= /opt/cuda

CXX := g++
CC := gcc
LINK := gcc -fPIC

INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) -arch=sm_20

NVCC := nvcc
LINKLINE = nvcc -o $@ ${PROGRAM_OBJS} $(LIB_CUDA) $(LIBS)

#--------------------------


# Libraries to include (if any)
LIBS=-lm -lrt

# Compiler flags
CFLAGS=-Wall -W -g -Wmissing-prototypes

# Indentation flags
IFLAGS=-br -brs -npsl -ce -cli4

# Name of the executable
PROGRAM=sortXXL

# Prefix for the gengetopt file (if gengetopt is used)
PROGRAM_OPT=cmdline

# Object files required to build the executable
PROGRAM_OBJS=sortXXL.o debug.o memory.o mregex.o io.o glist.o cudaEngine.o controller.o ${PROGRAM_OPT}.o benchmark.o demo.o

# Clean and all are not files
.PHONY: clean all docs indent

all: ${PROGRAM}

# compilar com depuracao
debugon: CFLAGS += -D SHOW_DEBUG -g
debugon: ${PROGRAM}

${PROGRAM}: ${PROGRAM_OBJS}
	$(LINKLINE)	

# Dependencies
sortXXL.o: sortXXL.cu debug.h memory.h mregex.h io.h glist.h controller.h ${PROGRAM_OPT}.h cudaEngine.h

${PROGRAM_OPT}.o: ${PROGRAM_OPT}.c ${PROGRAM_OPT}.h

glist.o: glist.cu glist.h 
io.o: io.cu io.h memory.h glist.h controller.h
controller.o: controller.cu controller.h memory.h glist.h io.h benchmark.h demo.h cudaEngine.h
mregex.o: mregex.cu mregex.h
memory.o: memory.cu memory.h
debug.o: debug.cu debug.h
benchmark.o: benchmark.cu benchmark.h
demo.o: demo.cu demo.h
cudaEngine.o: cudaEngine.cu cudaEngine.h

.SUFFIXES: .c .cpp .cu .o
#how to create an object file (.o) from C file (.c)
.c.o:
	${CC} ${CFLAGS} -c $<

.cu.o:
	$(NVCC) $(NVCCFLAGS) -c $< 


# Generates command line arguments code from gengetopt configuration file
${PROGRAM_OPT}.h: ${PROGRAM_OPT}.ggo
	gengetopt --input=${PROGRAM_OPT}.ggo --file-name=${PROGRAM_OPT} -a cmdline_args_info --func-name=cmdline_parser

clean:
	rm -f *.o core.* *~ ${PROGRAM} *.bak ${PROGRAM_OPT}.h ${PROGRAM_OPT}.cu

docs: Doxyfile
	doxygen Doxyfile

Doxyfile:
	doxygen -g Doxyfile

indent:
	indent ${IFLAGS} *.c *.h
