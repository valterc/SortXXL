#ifndef __MREGEX_H_
#define __MREGEX_H_

/**
 * Checks if a string is a valid pattern using regex
 * @param pattern Regex pattern
 * @param string String to match
 * @return 0 if not match, 1 if match
 */
int ismatch(const char *string, char *pattern);

#endif /* __MREGEX_H_ */
