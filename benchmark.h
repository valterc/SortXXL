#ifndef BENCHMARK_H
#define BENCHMARK_H

#define ERR_B_NO_ERROR		 0x000
#define ERR_B				-0x001

/** @brief Type that olds all the statistics */
typedef struct{
	float stdd; /**< @brief Standard deviation*/
	float med; /**< @brief Average*/
	float min; /**< @brief Minimum execution time*/
	float max; /**< @brief Maximum execution time*/
    float *times; /**< @brief Execution times*/
	int nt; /**< @brief Execution count*/
}Statistics;

/**
 * Initializes the benchmark
 * @param runTimes Number of time that the sort will run 
 */
void BENCH_init(int runTimes);

/**
 * Registers a run
 * @param runCount Number of the run
 * @param time Elapsed time in the run
 */
void registerRun(int runCount, float time);

/**
 * Generates statistics
 * @return Calculated statistics
 */
Statistics *generateStatistics(void);

/**
 * Return the current statistics
 * @return Calculated statistics
 */
Statistics *getStatistics(void);

/**
 * Disposed data from benchmark
 */
void BENCH_dispose(void);

#endif
