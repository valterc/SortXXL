#ifndef GLIST_H
#define GLIST_H

/** @brief List Element */
typedef struct g_list_elem{
	void *value; 				/**< @brief Value  */
	struct g_list_elem *next;	/**< @brief Next element  */
}GListElem;

/** @brief Generic Linked List */
typedef struct{
	GListElem *first;			/**< @brief First element of the list  */
	GListElem *last;			/**< @brief Last element of the list  */
	unsigned int count;			/**< @brief Element count  */
}GList;

/**
 * Initializes a new linked list
 * @return Empty linked list
 */
GList *GL_init(void);

/**
 * Adds a item to the list
 * @param list Linked list
 * @param value Value to add
 */
void GL_add(GList *list, void *value);

/**
 * Adds a int value to the list
 * @param list Linked list
 * @param value Value to add
 */
void GL_addInt(GList *list, int value);

/**
 * Get the value at a certain position of the list
 * @param list Linked list
 * @param index The index of the value
 * @return value 
 */
void *GL_valueAt(GList *list, unsigned int index);

/**
 * Releases the resources used by the list
 * @param list Linked list
 * @param f The function to release the elements resources
 */
void GL_free(GList *list, void (*f)(void *));

/**
 * Runs the function f for each element of a list
 * @param list Linked list
 * @param f The function to be inxoked for each element
 */
void GL_foreach(GList *list, void (*f)(void *));

#endif
