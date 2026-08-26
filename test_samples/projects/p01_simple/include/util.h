#ifndef UTIL_H
#define UTIL_H
#include <stddef.h>

void copy_data(char *dst, const char *src);
int  checksum(const char *s, size_t n);

#endif
