#include <stdio.h>

static void helper(int x) {
    printf("B%d\n", x);
}

void run_b(int x) {
    helper(x);
}
