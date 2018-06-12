#ifndef DEMO_H
#define DEMO_H

#define ERR_D_NO_ERROR		 0x000
#define ERR_D				-0x001

#define STAGE 				"<table><tbody><tr class=\"rows\"> <th scope=\"row\">First stage</th> <td>%f</td> </tr><tr class=\"rows\"> <th scope=\"row\">Second stage</th> <td>%f</td> </tr><tr class=\"rows\"> <th scope=\"row\">Third stage</th> <td>%f</td> </tr><tr class=\"rows\"> <th scope=\"row\">Fourt stage</th> <td>%f</td> </tr></tbody></table>"


#define BENCHMARK_TABLE_BEGIN		"<table id=\"data\"><tfoot><tr>\n"
#define BENCHMARK_VALUE_1			"<th>%d</th>\n"
#define BENCHMARK_TABLE_MIDDLE			"</tr></tfoot><tbody><tr>\n"
#define BENCHMARK_VALUE_2			"<td>%f</td>\n"
#define BENCHMARK_TABLE_END			"</tr></tbody></table>\n"


#define NUMBERS_BEGIN		"<h2>Sorted Numbers</h2><div class=\"small-box\">"
#define NUMBERS_NUM			"%d <br>"
#define NUMBERS_END			"</div>"

#include "controller.h"

/**
 * Initializes the demo
 */
void DEMO_init(void);

/**
 * Starts the demo operation
 */
void DEMO_startRun(void);

/**
 * New stage alert
 */
void DEMO_alertNewStage(int fase);

/**
 * Alerts of the completition of the sort
 */
void DEMO_completeExecution(SortResult *result);

/**
 * Disposed all resources allocated by Demo
 */
void DEMO_dispose(void);

#endif
