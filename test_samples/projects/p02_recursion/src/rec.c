#include <stdio.h>

int fact(int n) {
    if (n <= 1) return 1;
    return n * fact(n - 1);
}

static int ping(int n);
static int pong(int n);

static int ping(int n) {
    if (n <= 0) return 0;
    return pong(n - 1);
}

static int pong(int n) {
    if (n <= 0) return 1;
    return ping(n - 1);
}

void driver(int k) {
    printf("%d %d\n", fact(k), ping(k));
}
