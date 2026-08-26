#include <string.h>

int transform(int acc, int n) {
    int i = 0;
    for (i = 0; i < n; i++) {
        acc += i;
    }
    return acc;
}

int caller(int n) {
    int acc = 100;
    int i = 7;
    acc = transform(acc, i);
    return acc + i;
}
