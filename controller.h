#ifndef CONTROLLER_H
#define CONTROLLER_H

#define ERR_C_NO_ERROR		 0x000
#define ERR_C				-0x001
#define ERR_C_INVALID_INPUT	-0x002
#define BILLION  1E9;

#include "benchmark.h"

/** @brief Numbers holder */
typedef struct{
	int *numbers; /** @brief Numbers */
	unsigned int count; /** @brief Numbers count */
}Numbers;


/** @brief Holds the results of the sorts operation */
typedef struct{
	int *sortedNumbers; /** @brief Sorted numbers */
	unsigned int numbersCount; /** @brief Numbers count */
	unsigned int padCount; /** @brief Padding count */
	Statistics *benchmarkStatistics; /** @brief Statistics */
	float elapsedGPUTime; /** @brief Elapsed GPU time */
	float elapsedRealTime; /** @brief Elapsed Real time */
}SortResult;

/**
 * Initializes the controller
 */
void CTRL_init(void);

/**
 * Disposes all resources allocated by controller
 */
void C_dispose(void);

/**
 * Set the numbers to be sorted
 * @param fromfile Load numbers from file
 * @param filePath File path
 * @param randomCount Count of random numbers
 * @param min Minimum random number
 * @param max Maximum random number
 * @return 0 on success, otherwise error
 */
int setNumbers(int fromFile, char* filePath, unsigned int randomCount, int min, int max);

/**
 * Sorts the numbers
 * @param benchmark 
 * @param demo
 * @return The result of the sort operation
 */
SortResult *sort(int benchmark, int demo);

/**
 * Printf the sort information to screen
 * @param result Sort results
 * @param printgpu Flag to print gpu info
 */
void printResultsToScreen(SortResult *result, int printgpu);


#endif
