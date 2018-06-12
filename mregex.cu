/**
* @file mregex.c
* @author Valter Costa
* @date 28-10-2010
* @brief This file contains usefull REGEX functions
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "mregex.h"
#include <regex.h>

int ismatch(const char *string, char *pattern) 
{
	int stat; 
	regex_t re;
 
	if(regcomp(&re, pattern, REG_EXTENDED|REG_NOSUB|REG_ICASE) != 0) 
	{
		return 0; 
	}
 
	stat = regexec(&re, string, (size_t)0, NULL, 0);
	regfree(&re);
 

	if(stat == REG_NOMATCH) 
	{
		return 0; 
    }
 
	return 1; 
}
