/**
* @file glist.cu
* @brief Linked list implementation
* @date 03-12-2012
* @author Valter Costa 2120908@my.ipleiria.pt
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

#include "glist.h"

// -------------- STATIC FUNCTIONS


// -------------- END STATIC FUNCTIONS

GList *GL_init(void){
	GList *list = (GList *)malloc(sizeof(GList));
	memset(list, 0, sizeof(GList));
	return list;
}

void GL_add(GList *list, void *value){
	GListElem *elem = (GListElem *)malloc(sizeof(GListElem));
	memset(elem, 0, sizeof(GListElem));
	elem->value = value;
	
	if (!list->last) {
		list->first = elem;
		list->last = elem;
	} else {
		list->last->next = elem;
		list->last = elem;
	}
	list->count++;
}

void GL_addInt(GList *list, int value){
	GListElem *elem = (GListElem *)malloc(sizeof(GListElem));
	memset(elem, 0, sizeof(GListElem));
	
	elem->value = (int *)malloc(sizeof(int));
	*(int *)elem->value = value;
	
	if (!list->last) {
		list->first = elem;
		list->last = elem;
	} else {
		list->last->next = elem;
		list->last = elem;
	}
	list->count++;
}

void * GL_valueAt(GList *list, unsigned int index)
{
	if (index > list->count -1)
		return NULL;
	
	//TODO: if index is closer to count value than should start from end
	
	GListElem *elem = list->first;
	unsigned int i = 0;
	for (i = 0; i < index; i++)
	{
		elem = elem->next;
	}
	
	return elem->value;
}

void GL_free(GList *list, void (*f)(void *))
{
	GListElem *elem = list->first;
	GListElem *nextElem = NULL;
	while (elem != NULL)
	{
		if (f)
			f(elem->value);
		
		nextElem = elem->next;
		free(elem);
		elem = nextElem;
	}
	
	free(list);
}

void GL_foreach(GList *list, void (*f)(void *))
{
	if (list == NULL)
		return;
	
	GListElem *elem = list->first;
	while (elem != NULL)
	{
		f(elem->value);
		elem = elem->next;
	}
	
}
