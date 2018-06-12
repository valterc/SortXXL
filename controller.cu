/**
* @file controller.c
* @brief Handles the main logic of the sortXXL program
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
#include <limits.h>
#include <sys/time.h>


#include "io.h"
#include "glist.h"
#include "memory.h"
#include "controller.h"
#include "benchmark.h"
#include "cudaEngine.h"
#include "demo.h"

extern int G_verbose;
Numbers *G_numbers;
struct timeval startcpu, stopcpu;
SortResult *computeResult;

// -------------- STATIC FUNCTIONS
/**
 * Initializes the random generator seed
 */
static void initRandomSeed(void){
	srandom(time(NULL));
}

/**
 * Generates count random numbers between min and max
 * @param count Number count
 * @param min Minimum value
 * @param max Maximum value
 * @return Random numbers
 */
static Numbers *generateRandomNumbers(unsigned int count, double min, double max){
	initRandomSeed();
	
	Numbers *numbers = (Numbers *)malloc(sizeof(Numbers));
	memset(numbers, 0, sizeof(Numbers));
	numbers->count = count;
	
	numbers->numbers = (int *)malloc(sizeof(int) * count);
	
	min = min - 1;
	
	unsigned int i;		
	for (i = 0; i < count; i++)
	{
		//generate a random place in a [0;1[ interval
		double n = (double)random()/(RAND_MAX + 1.0);
		
		//uses the generated place to find a number in the same place but this time in ]min;max[ interval
		int val = ((max - min + 1) * n) + min;
		
		//just to make sure
		if (val < min || val > max)
		{
			i--;
			continue;
		}
		
		numbers->numbers[i] = val;
	}
	
	return numbers;
}

/**
 * Copies the numbers in 'numbers' and fill until return is a 1024 multiplier
 * @param numbers Numbers to copy
 * @param count Count of numbers
 * @param max Value to fill
 * @return Copied numbers
 */
static Numbers *copyNumbersAndFillToTop(int *numbers, int count, int max){

	//We need to make sure that the number count is disible by 1024
	int newCount = count;
	int r = newCount % 1024;
	if (r != 0){
		if (G_verbose)
			printf("Specified value is not divisible by 1024, filling with %d numbers\n", 1024 - (newCount % 1024));
			
		newCount = newCount + 1024 - (newCount % 1024);
	} else if(newCount < 1024) {
		if (G_verbose)
			printf("Specified value is not divisible by 1024, filling with %d numbers\n", 1024 - newCount);
			
		newCount = 1024;
	}
	
	Numbers *n = (Numbers *)malloc(sizeof(Numbers));
	
	n->numbers = (int *)malloc(sizeof(int) * newCount);
	memcpy(n->numbers, numbers, sizeof(int) * count);
	
	int i;
	for (i = count; i < newCount; i++)
	{
		n->numbers[i] = max;
	}
	
	n->count = newCount;
	
	return n;
}

/**
 * Finds the maximum number in a array
 * @param numbers Array of numbers
 * @param count Count of numbers
 * @return Maximum value
 */
static int findMax(int *numbers, int count){
	
	int max = INT_MIN;
	
	int i;
	for (i = 0; i < count; i++)
	{
		if (numbers[i] > max)
			max = numbers[i];
	}
	
	return max;
}

/**
 * Returns the number of digits in a number
 * @param number Number
 * @return digits of the number
 */
static int numDigits(int number)
{
    int digits = 0;
    if (number == 0) return 1;
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}

// -------------- END STATIC FUNCTIONS

void CTRL_init(void){
	gettimeofday(&startcpu, NULL);
}

void C_dispose(void)
{
	if (G_numbers != NULL){
		free(G_numbers->numbers);
		free(G_numbers);
	}
}

int setNumbers(int fromFile, char* filePath, unsigned int randomCount, int min, int max)
{
	if (fromFile) {
		G_numbers = (Numbers *)readInputFile(filePath);
		if (G_numbers == NULL || G_numbers->numbers == NULL || G_numbers->count < 2)
			return ERR_C_INVALID_INPUT;
		
		if (G_verbose)
			printf("Input count: %u\n", G_numbers->count);
			
		return ERR_C_NO_ERROR;
	}
	
	if (max < min || max == min || randomCount < 2)
		return ERR_C_INVALID_INPUT;
	
	
	G_numbers = generateRandomNumbers(randomCount, (double)min, (double)max);
	
	/*
	unsigned int i;
	for (i = 0; i < G_numbers->count; i++)
	{
		printf("%d\n", G_numbers->numbers[i]);
	}
	*/
	
	return 0;
}

SortResult *sort(int benchmark, int demo){
	
	if(G_verbose)
		printf("Initializing sort...\n");
	
	if (demo){
		if(G_verbose)
			printf("Initializing demo...\n");
			
		DEMO_init();
	}
		
	int maxValue = findMax(G_numbers->numbers, G_numbers->count);
	if(G_verbose)
			printf("Max value is %d\n", maxValue);
			
	int maxDigitCount = numDigits(maxValue);
	if(G_verbose)
			printf("Max digit count is %d\n", maxDigitCount);
			
	if (benchmark){
		if(G_verbose)
			printf("Benchmark mode\n");
			
		BENCH_init(benchmark);

		Numbers *numbers = copyNumbersAndFillToTop(G_numbers->numbers, G_numbers->count, maxValue);
		SortRunResult *sortResult = NULL;
		
		int i;
		for (i = 0; i < benchmark; i++)
		{
			if (sortResult != NULL) {
				free(sortResult->sortedNumbers);
				free(sortResult);
			}
			sortResult = radix_sort(numbers->numbers, numbers->count, maxDigitCount);
			registerRun(i, sortResult->elapsedTime);
		}
		
		computeResult = (SortResult *)malloc(sizeof(SortResult));
		computeResult->sortedNumbers = sortResult->sortedNumbers;
		computeResult->numbersCount = G_numbers->count;
		computeResult->benchmarkStatistics = generateStatistics();
		computeResult->elapsedGPUTime = sortResult->elapsedTime;
		computeResult->padCount = numbers->count - G_numbers->count;
		
		
		free(numbers->numbers);
		free(numbers);
		free(sortResult);
		
	} else {
		if(G_verbose)
			printf("One run mode\n");
		
		Numbers *numbers = copyNumbersAndFillToTop(G_numbers->numbers, G_numbers->count, maxValue);		
		SortRunResult *sortResult = radix_sort(numbers->numbers, numbers->count, maxDigitCount);		
		
		computeResult = (SortResult *)malloc(sizeof(SortResult));
		computeResult->sortedNumbers = sortResult->sortedNumbers;
		computeResult->numbersCount = G_numbers->count;
		computeResult->benchmarkStatistics = NULL;
		computeResult->elapsedGPUTime = sortResult->elapsedTime;
		computeResult->padCount = numbers->count - G_numbers->count;
		
		free(sortResult);
		free(numbers->numbers);
		free(numbers);
	}
	
	
	
	if (computeResult != NULL) {
		gettimeofday(&stopcpu, NULL);
		
		long seconds  = stopcpu.tv_sec  - startcpu.tv_sec;
		long useconds = stopcpu.tv_usec - startcpu.tv_usec;
		
		float time = (seconds + useconds * 1E-6);
		computeResult->elapsedRealTime = time;
	}
	
	DEMO_completeExecution(computeResult);
	DEMO_dispose();
	
	return computeResult;
}

void printResultsToScreen(SortResult *result, int printgpu){
	
	char *date = getDateTimeString();
	
	printf("SORTXXL Sort Complete\n");
	printf("Executed at: \t\t%s\n", date);
	printf("Total Execution time: \t%.5f s\n", result->elapsedRealTime);
	if (result->benchmarkStatistics == NULL)
		printf("GPU Execution time: \t%.5f ms\n", result->elapsedGPUTime);
		
	printf("Sorted numbers: \t%d (with %d numbers for padding)\n", result->numbersCount, result->padCount);
	
	if (printgpu){
		printGPUInfo();
	}
	
	if (result->benchmarkStatistics != NULL){
		printf("BENCHMARK----------\n");
		int x;
		for (x = 0; x < result->benchmarkStatistics->nt; x++)
		{
			printf("Execution %d: \t\t%.5f ms\n", x, result->benchmarkStatistics->times[x]);
		}
		
		printf("Max execution time: \t%.5f ms\n", result->benchmarkStatistics->max);
		printf("Min execution time: \t%.5f ms\n", result->benchmarkStatistics->min);
		printf("Average: \t\t%.5f ms\n", result->benchmarkStatistics->med);
		printf("Standard deviation: \t%.5f\n", result->benchmarkStatistics->stdd);
		printf("-------------------\n");
	}
	
	printf("First five numbers:\n");
	
	int i;
	for (i = 0; i < result->numbersCount && i < 5; i++)
	{
		printf("%d\n", result->sortedNumbers[i]);
	}
	
	printf("Last five numbers:\n");
	for (i = result->numbersCount - 5; i < result->numbersCount && i > -1; i++)
	{
		printf("%d\n", result->sortedNumbers[i]);
	}
	
	free(date);
}

