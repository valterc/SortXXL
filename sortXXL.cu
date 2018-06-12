/**
* @file sortXXL.c
* @brief SortXXL main file
* @date 26-11-2012
* @author Valter Costa, João Sousa, João Órfão {2120908,2120903,2120904}@my.ipleiria.pt
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <libgen.h>
#include <stdarg.h>
#include <limits.h>

#include "io.h"
#include "debug.h"
#include "memory.h"
#include "cmdline.h"
#include "controller.h"
#include "cudaEngine.h"

#define VERSION 1.0
#define CG_ABOUT				"\n:: SortXXL version 1 - 2013 ::\n"\
								"GPU Number sorter, made by:\n"\
								"Valter Costa\t--\t2120908@my.ipleiria.pt\n"\
								"João Sousa\t--\t2120903@my.ipleiria.pt\n"\
								"João Órfão\t--\t2120904@my.ipleiria.pt\n"\
								"High Performace Computing, Masters of Computer Engineering - Mobile Computing\n\n"

//reinvent the wheel just because
#define ASS(F,...) int r = F; if (r!=0) { disposeAll(); cmdline_parser_free(&args_info); printErrorAndDie(r,__VA_ARGS__); return r;}

// ------------------- 
void printErrorAndDie(int err, const char *s, ...);

// ------------------- STATIC FUNCTIONS

/**
 * Disposes all resources
 */
static void disposeAll(void){
	IO_dispose();
	C_dispose();
	BENCH_dispose();
}

// ------------------- END STATIC FUNCTIONS


// ------------------- GLOBALS
int G_verbose;
char *G_appName;

// ------------------- END GLOBALS



int main(int argc, char *argv[])
{
	G_appName = argv[0];

	(void)argc; (void)argv;

	struct cmdline_args_info args_info;
	cmdline_parser_init(&args_info);
	
	if (cmdline_parser(argc,argv,&args_info) != 0)
	{
		printErrorAndDie(-1, "Parsing arguments\n");
	}

	if (cmdline_parser(argc,argv,&args_info) != 0)
	{
		printErrorAndDie(-1, "Parsing arguments\n");
	}

	if (args_info.about_given)
	{
		printf(CG_ABOUT);
		cmdline_parser_free(&args_info);
		return 0;
	}

	if (args_info.input_given == 1 && args_info.random_given == 1)
	{
		printErrorAndDie(-1, "'--input' ('-i') and '--random' ('-r') options cannot be specified at the same time\n");
	}
	
	if (args_info.input_given == 0 && args_info.random_given == 0)
	{
		if (args_info.gpu_given) {
			printGPUInfo();
			cmdline_parser_free(&args_info);
			return 0;
		}
		printErrorAndDie(-1, "'--input' ('-i') or '--random' ('-r') must be specified\n");
	}

	if (args_info.max_given == 0)
	{
		args_info.max_arg = INT_MAX;
	}
	
	if (args_info.min_given == 0)
	{
		args_info.min_arg = INT_MIN;
	}

	if (args_info.max_arg < args_info.min_arg)
		printErrorAndDie(-1, "'--max' ('-M') cannot be lower than '--min' ('-m')\n");

	G_verbose = args_info.verbose_given;
	
	CTRL_init();
	
	ASS(
		setNumbers(args_info.input_given, args_info.input_arg, args_info.random_arg, args_info.min_arg, args_info.max_arg), 
		"Invalid input given. \nProvide a valid file or use valid random numbers"
	)
	
	
	SortResult *result = sort(args_info.benchmark_arg, args_info.demo_given);

	printResultsToScreen(result, args_info.gpu_given);
	

	if (args_info.output_given) {
		
		ASS(
			saveOutputFile(args_info.output_arg, result, args_info.gpu_given), 
			"Error writing results to output file"
		)
		
	}
	

	disposeAll();
	
	cmdline_parser_free(&args_info);

	if (G_verbose)
		printf("All done.\n");

	return 0;
}


/**
 * Prints a formatted error message to stdout and exits
 * @param err Error code
 * @param s Formatted message to be written
 * @param ... optional parameters
 */
void printErrorAndDie(int err, const char *s, ...)
{
	va_list argp;
	va_start(argp, s);
	
	printf("[%s|%d] Error: ", G_appName, getpid());
	
	if (s != NULL && strlen(s) > 0){
		vprintf(s, argp);
		if (s[strlen(s)-1] != '\n')
			printf("\n");
	} else {
		printf("Application terminated unexpectedly\n");
	}
	
	fflush(stdout);
	va_end(argp);
	exit(err);
}

