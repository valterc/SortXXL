#include "memory.h"
#include <stdio.h>


void *mmalloc(size_t size, const int line, const char *file) {
	
	void *ptr = malloc(size);

	if( ptr == NULL ) {
		fprintf(stderr, "[%d@%s][ERROR] can't malloc %u bytes\n",line, file, size);
	}
		
	return ptr;
}

void *mrealloc(void *ptr, size_t size, const int line, const char *file) {
	
	ptr = realloc(ptr, size);

	if( ptr == NULL ) {
		fprintf(stderr, "[%d@%s][ERROR] can't realloc %u bytes\n",line, file, size);
	}
		
	return ptr;
}

void mfree(void **ptr, const int line, const char *file) {
	(void)line;
	(void)file;
	free(*ptr);
	*ptr = NULL;
}
