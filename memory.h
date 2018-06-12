#ifndef MEMORY_H
#define MEMORY_H
#include <stdlib.h>
#define MMALLOC(size) mmalloc((size),__LINE__,__FILE__);
#define MREALLOC(ptr, size) mrealloc((ptr),(size),__LINE__,__FILE__);
#define MFREE(ptr) mfree((void**)(&(ptr)),__LINE__,__FILE__);

void *mmalloc(size_t size, const int line, const char *file);
void *mrealloc(void *ptr, size_t size, const int line, const char *file);
void mfree(void **ptr, const int line, const char *file);

#endif
