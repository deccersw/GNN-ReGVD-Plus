#include <stdio.h>

static void helper(int x) {
    printf("A%d\n", x);
}

void run_a(int x) {
    helper(x);
}
