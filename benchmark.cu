/**
* @file benchmark.c
* @brief Handles time benchmarking
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

#include "benchmark.h"

Statistics statistic;

void BENCH_init(int runTimes){
	statistic.nt = runTimes;
	statistic.min = 0;
	statistic.max = 0;
	statistic.med = 0;
	statistic.stdd = 0;
	statistic.times = (float *)malloc(sizeof(float) * runTimes);
}

void registerRun(int runCount, float time){
	statistic.times[runCount] = time;
}

Statistics *generateStatistics(void){
	Statistics *stat = &statistic;
	
	stat->min = stat->times[0];
	
	int aux1;
	for(aux1 = 0; aux1<stat->nt; aux1++){
		if(stat->times[aux1]<stat->min)
			stat->min = stat->times[aux1];
		
		if(stat->times[aux1]>stat->max)
			stat->max = stat->times[aux1];
		
		stat->med += stat->times[aux1];
	}
	
	stat->med = stat->med/stat->nt;

	int aux2;
	for(aux2 = 0; aux2<stat->nt; aux2++){

		stat->stdd += (stat->times[aux2] - stat->med)*(stat->times[aux2] - stat->med);	
	}
	
	stat->stdd = sqrt(stat->stdd/(stat->nt-1));
	
	return stat;
}

Statistics *getStatistics(void){
	return &statistic;
}

void BENCH_dispose(void){
	free(statistic.times);
}
