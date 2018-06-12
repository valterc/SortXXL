/**
* @file io.c
* @brief Handles any file/directory operation
* @date 03-12-2012
* @author Valter Costa, João Sousa, João Órfão {2120908,2120903,2120904}@my.ipleiria.pt
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <libgen.h>
#include <stdarg.h>

#include "io.h"
#include "glist.h"
#include "memory.h"
#include "controller.h"
#include "cudaEngine.h"

extern int G_verbose;

// -------------- STATIC FUNCTIONS
/**
 * Printf a int to stdout
 */
/*
static void pNumber(void *n){
	printf ("%d\n", *(int *)n);
}
*/

/**
 * Gets the current DateTime
 * @return DateTime string
 */
char *getDateTimeString(void) 
{
	time_t t;
	struct tm *tmp;
	t = time(NULL);
	tmp = localtime(&t);

	char *time = (char *)MMALLOC(sizeof(char) * 50);
	strftime(time, 50, "%Y_%m_%d %H:%M:%S", tmp);
	//printf("%s\n", time);
	
	return time;
}

// -------------- END STATIC FUNCTIONS


int writeOnFile(char *path, char *text)
{
	FILE *file;
	file = fopen(path,"a");
	
	if(file==NULL)
	{
		return ERR_IO;
	}
	
	int result = fprintf(file,"%s",text);
	fclose(file);

	return result < 1 ? ERR_IO : 0;
}

int writeOnFileArgs(char *path, char *str, ...)
{
	va_list argp;
	va_start(argp, str);
	
	FILE *file;
	file = fopen(path,"w");
	
	if(file==NULL)
	{
		return ERR_IO;
	}
	
	int result = vfprintf(file, str, argp);
	fclose(file);
	
	va_end(argp);
	
	return result;
}

void IO_dispose(void)
{
	
}

int fileExists(char *path)
{
	if(path != NULL){
		FILE *fptr = fopen(path, "r");
		if(fptr){
			fclose(fptr);
			return 1;
		}else{
			return 0;
		}
	}else{
		return 0;
	}
}

void *readInputFile(char * filePath)
{
	FILE *f = fopen(filePath, "r");
	if (f == NULL)
		return NULL;
	
	GList *list = GL_init();
	
	int number = 0;
	while(fscanf(f, "%d[^ ]", &number) != EOF )
	{
		GL_addInt(list, number);
	}
	fclose(f);
	
	Numbers *numbers =  (Numbers *)malloc(sizeof(Numbers));
	
	int *nums = (int *)malloc(sizeof(int) * list->count);
	unsigned int i = 0;
	GListElem *elem = list->first;
	for (i = 0; i < list->count; i++)
	{
		nums[i] = *(int *)elem->value;
		elem = elem->next;
	}
	
	//GL_foreach(list, pNumber);
	
	numbers->numbers = nums;
	numbers->count = list->count;
	
	GL_free(list, free);
	return numbers;
}

int saveOutputFile(char * filep, SortResult *result, int infogpu){
	
	char *date = getDateTimeString();
	char *filePath = strdup(filep);
	
	if (fileExists(filePath)) {
		free(filePath);
		filePath = (char *)malloc(sizeof(char) * (strlen(filep) + strlen(date) + 20));
		memset(filePath, 0, sizeof(char) * (strlen(filep) + strlen(date) + 20));
		strcat(filePath, filep); 
		strcat(filePath, "__");
		strcat(filePath, date);
	}
	
	FILE *file;
	file = fopen(filePath,"w");
	
	if(file==NULL)
	{
		return ERR_IO;
	}

	fprintf(file, "#SORTXXL - GPU NUMBER SORTER\n#\n");	
	fprintf(file, "#Executed at: \t\t\t%s\n", date);
	fprintf(file, "#Total Execution time: \t%.5f s\n", result->elapsedRealTime);
	
	if (result->benchmarkStatistics == NULL)
		fprintf(file, "#GPU Execution time: \t%.5f ms\n", result->elapsedGPUTime);
		
	fprintf(file, "#Sorted numbers: \t\t%d (with %d numbers for padding)\n\n", result->numbersCount, result->padCount);
	
	if (infogpu){
		printGPUInfoToFile(file);
	}
	
	if (result->benchmarkStatistics != NULL){
		fprintf(file, "BENCHMARK----------\n");
		int x;
		for (x = 0; x < result->benchmarkStatistics->nt; x++)
		{
			fprintf(file, "#Execution %d: \t\t\t%.5f ms\n", x, result->benchmarkStatistics->times[x]);
		}
		
		fprintf(file, "#Max execution time: \t%.5f ms\n", result->benchmarkStatistics->max);
		fprintf(file, "#Min execution time: \t%.5f ms\n", result->benchmarkStatistics->min);
		fprintf(file, "#Average: \t\t\t\t%.5f ms\n", result->benchmarkStatistics->med);
		fprintf(file, "#Standard deviation: \t%.5f\n", result->benchmarkStatistics->stdd);
		fprintf(file, "-------------------\n\n\n");
	}
	
	unsigned int i;
	for (i = 0; i < result->numbersCount; i++)
	{
		fprintf(file, "%d\n", result->sortedNumbers[i]);
	}
	
	
	fclose(file);
	
	free(date);
	free(filePath);
	
	return 0;
}
