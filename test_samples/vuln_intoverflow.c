/* CWE-190: Integer Overflow leading to Heap Overflow */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void copy_data(int count, const char *src) {
    int size = count * sizeof(int);  /* integer overflow if count is large */
    char *buf = malloc(size);
    if (buf) {
        memcpy(buf, src, count * sizeof(int));  /* writes more than allocated */
        printf("Copied %d items\n", count);
        free(buf);
    }
}

int main() {
    char data[4096];
    memset(data, 'B', sizeof(data));
    copy_data(1073741824, data);  /* INT_MAX/2: causes overflow in size calc */
    return 0;
}
