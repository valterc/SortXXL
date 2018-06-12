#ifndef CUDAENGINE_H
#define CUDAENGINE_H

#define ERR_CE_NO_ERROR		 0x000
#define ERR_CE				-0x001

/** @brief Holds the result of a sort run */
typedef struct {
	int *sortedNumbers; /**< @brief Array containing the sorted numbers */
	float elapsedTime; /**< @brief Array containing the sorted numbers */
}SortRunResult;

/** @brief Holds the info a GPU */
typedef struct{
	char *Name; /**< @brief Name of the GPU */
	int versionMajor; /**< @brief Major version */
	int versionMinor; /**< @brief Minor version */
	int clockRate; /**< @brief Clock rate */
	unsigned int globalMemory; /**< @brief Global memory value */
	unsigned int constantMemory; /**< @brief Constant memory value */
	int multiprocessorCount; /**< @brief Multiprocessor count */
	int cudaCores; /**< @brief Cuda cores */
}GPUInfo;

/**
 * Sorts the numbers in GPU
 * @param numbers Numbers to be sorted
 * @param numbersCount Count of the numbers to be sorted
 * @param digitCount The digit count of the maximum number
 * @return Sort result
 */
SortRunResult *radix_sort(int *numbers, int numbersCount, int digitCount);

/**
 * Prints the information of the gpu
 */
void printGPUInfo();

/**
 * Prints the information of the gpu to a file 
 * @param f File
 */
void printGPUInfoToFile(FILE *f);

/**
 * queries the GPU info to a GPUInfo struct
 */
GPUInfo* queryGPUInfo();

#endif
