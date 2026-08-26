#include "util.h"
#include <string.h>

void copy_data(char *dst, const char *src) {
    strcpy(dst, src);
}

int checksum(const char *s, size_t n) {
    int acc = 0;
    for (size_t i = 0; i < n; i++) {
        acc += s[i];
    }
    return acc;
}
