#ifndef IO_H
#define IO_H

#define ERR_IO				-0x001

#include "controller.h"

/**
 * Disposes all resources allocated by IO
 * 
 */
void IO_dispose(void);

/**
 * Append text to a file, if the file does not exist it is created
 * @param path File path
 * @param text String to append to the file
 * @return 0 if succeded, otherwise error
 */
int writeOnFile(char *path, char *text);

/**
 * Append text to a file, if the file does not exist it is created
 * @param path File path
 * @param str String to append to the file
 * @param args
 * @return number of bytes written to file
 */
int writeOnFileArgs(char *path, char *str, ...);

/**
 * Checks if a file exists
 * @param path [IN] Location of the file
 * @return 1 if the file does exist, 0 otherwise
 */
int fileExists(char *path);

/**
 * Reads a input file and processed all the numbers
 * @param filePath Path of the file
 */
void *readInputFile(char * filePath);

/**
 * Saves the sorted information to a file
 * @param filep File Path
 * @param result Sort result
 * @param infogpu Flag to print gpu information
 */
int saveOutputFile(char * filep, SortResult *result, int infogpu);

/**
 * Gets the current Date/time
 * @return String representing the current DateTime
 */
char *getDateTimeString(void);


#endif
