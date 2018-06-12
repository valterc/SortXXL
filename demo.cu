/**
* @file demo.cu
* @brief Handles the demo feature
* @date 07-01-2013
* @author Valter Costa, João Órfão, João Sousa,  {2120908, 2120904, 2120903}@my.ipleiria.pt
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
#include <math.h>
#include <cuda.h>

#include "io.h"
#include "demo.h"
#include "cudaEngine.h"
#include "HandleError.h"




cudaEvent_t *events;
float *elapsedTimes;
int demoEnabled = 0;

static void writeFirstInfoToFile(void){
	writeOnFileArgs("demo/auxfiles/statistics.html", STAGE, 0, 0, 0, 0);
}

static void writeInfoToFile(int x, cudaEvent_t evt0, cudaEvent_t evt1){
	float elapsedTime = 0;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, evt0, evt1 ) );
	
	elapsedTimes[x] = elapsedTime;
	
	writeOnFileArgs("demo/auxfiles/statistics.html", STAGE, elapsedTimes[1], elapsedTimes[2], elapsedTimes[3], elapsedTimes[4]);
}



void DEMO_init(void){
	demoEnabled = 1;
	
	events = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 5);
	elapsedTimes = (float *)malloc(sizeof(float) * 5);
	
	int i;
	for (i = 0; i < 5; i++)
	{
		cudaEventCreate( &(events[i]) );
		elapsedTimes[i] = 0;
	}
	
	remove("demo/auxfiles/statistics.html");
	remove("demo/auxfiles/stat.html");
	remove("demo/auxfiles/bench.html");
	remove("demo/auxfiles/benchdata.html");
	remove("demo/auxfiles/execution.html");
	remove("demo/auxfiles/numbers.html");
	remove("demo/auxfiles/gpudetails.html");
	
	/*
	 * delete:
	 * bench.html
	 * statistics.html
	 * benchdata.html
	 * 
	 */
	
}

void DEMO_startRun(void){
	if (demoEnabled == 0)
		return;
		
	HANDLE_ERROR( cudaEventRecord( events[0], 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( events[0] ) );
	
	writeFirstInfoToFile();
}

void DEMO_alertNewStage(int stage){
	if (demoEnabled == 0)
		return;
		
	HANDLE_ERROR( cudaEventRecord( events[stage], 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( events[stage] ) );
	writeInfoToFile(stage, events[stage-1], events[stage]);
	
	writeOnFileArgs("demo/auxfiles/stat.html", "%d", stage);
}

void DEMO_completeExecution(SortResult *result){
	if (demoEnabled == 0)
		return;
		
	writeOnFileArgs("demo/auxfiles/stat.html", "5");
		
	int i;
	FILE *file;
		
	if (result->benchmarkStatistics != NULL){
		
		
		file = fopen("demo/auxfiles/bench.html","w");
		if(file!=NULL)
		{
			fprintf(file, BENCHMARK_TABLE_BEGIN);
			
			for (i = 0; i < result->benchmarkStatistics->nt; i++)
			{
				fprintf(file, BENCHMARK_VALUE_1, i);
			}
			
			
			fprintf(file, BENCHMARK_TABLE_MIDDLE);
			
			for (i = 0; i < result->benchmarkStatistics->nt; i++)
			{
				fprintf(file, BENCHMARK_VALUE_2, result->benchmarkStatistics->times[i]);
			}
			
			fprintf(file, BENCHMARK_TABLE_END);
			fclose(file);
		}
	
		file = fopen("demo/auxfiles/benchdata.html","w");
		if(file!=NULL)
		{
			fprintf(file, "<p>Max execution time: <b>%.5f ms</b></p>", result->benchmarkStatistics->max);
			fprintf(file, "<p>Min execution time: <b>%.5f ms</b></p>", result->benchmarkStatistics->min);
			fprintf(file, "<p>Average: <b>%.5f ms</b></p>", result->benchmarkStatistics->med);
			fprintf(file, "<p>Standard deviation: <b>%.5f</b></p>", result->benchmarkStatistics->stdd);
			fclose(file);
		}
		
		
	}
	
	file = fopen("demo/auxfiles/numbers.html","w");
	fprintf(file, NUMBERS_BEGIN);
	for (i = 0; i < result->numbersCount; i++)
	{
		fprintf(file, NUMBERS_NUM, result->sortedNumbers[i]);
	}
	fprintf(file, NUMBERS_END);
	fclose(file);
		
	file = fopen("demo/auxfiles/execution.html","w");
	if (file != NULL){
		fprintf(file, "<h2>Execution details</h2><div class=\"small-box\">");
		
		char *date = getDateTimeString();
		
		fprintf(file, "<p>Executed at: <b>%s</b></p>", date);
		fprintf(file, "<p>Total Execution time: <b>%.5f s</b></p>", result->elapsedRealTime);
		
		if (result->benchmarkStatistics == NULL)
			fprintf(file, "<p>GPU Execution time: <b>%.5f ms</b></p>", result->elapsedGPUTime);
			
		fprintf(file, "<p>Sorted numbers: <b>%d</b> (with <b>%d</b> numbers for padding)</p>", result->numbersCount, result->padCount);
		fprintf(file, "</div>");
		
		free(date);
		fclose(file);	
	}
	
	GPUInfo *gpuInfo = queryGPUInfo();

	file = fopen("demo/auxfiles/gpudetails.html","w");
	if (file != NULL){
		fprintf(file, "<h2>GPU Information</h2><div class=\"small-box\">");
		
		fprintf(file, "<p>Name: <b>%s</b></p>", gpuInfo->Name);
		fprintf(file, "<p>Compute capability: <b>%d.%d</b></p>", gpuInfo->versionMajor, gpuInfo->versionMinor  );
		fprintf(file, "<p>Clock rate: <b>%d</b></p>", gpuInfo->clockRate );
		fprintf(file, "<p>Total global mem: <b>%zu</b></p>", gpuInfo->globalMemory );
		fprintf(file, "<p>Total constant Mem: <b>%zu</b></p>", gpuInfo->constantMemory );
		fprintf(file, "<p>Multiprocessor count: <b>%d</b></p>", gpuInfo->multiprocessorCount );
		fprintf(file, "<p>Cuda cores: <b>%d</b></p>", gpuInfo->cudaCores);
		
		
		fprintf(file, "</div>");
		
		fclose(file);	
	}

	free(gpuInfo->Name);
	free(gpuInfo);
}

void DEMO_dispose(void){
	if (demoEnabled == 0)
		return;
		
	int i;
	for (i = 0; i < 5; i++)
	{
		HANDLE_ERROR( cudaEventDestroy( events[i] ) );
	}
	
}


