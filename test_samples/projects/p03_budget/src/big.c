#include <string.h>

void tiny_helper(char *d, const char *s) {
    strcpy(d, s);
}

void huge_helper(int *out, int n) {
    int a0 = 0; int a1 = 1; int a2 = 2; int a3 = 3; int a4 = 4;
    int a5 = 5; int a6 = 6; int a7 = 7; int a8 = 8; int a9 = 9;
    int b0 = 0; int b1 = 1; int b2 = 2; int b3 = 3; int b4 = 4;
    int b5 = 5; int b6 = 6; int b7 = 7; int b8 = 8; int b9 = 9;
    int c0 = 0; int c1 = 1; int c2 = 2; int c3 = 3; int c4 = 4;
    int c5 = 5; int c6 = 6; int c7 = 7; int c8 = 8; int c9 = 9;
    int d0 = 0; int d1 = 1; int d2 = 2; int d3 = 3; int d4 = 4;
    int d5 = 5; int d6 = 6; int d7 = 7; int d8 = 8; int d9 = 9;
    int e0 = 0; int e1 = 1; int e2 = 2; int e3 = 3; int e4 = 4;
    int e5 = 5; int e6 = 6; int e7 = 7; int e8 = 8; int e9 = 9;
    for (int i = 0; i < n; i++) {
        out[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9
               + b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9
               + c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
               + d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9
               + e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8 + e9;
    }
}

void entry(const char *input, int *results, int n) {
    char local[32];
    huge_helper(results, n);
    tiny_helper(local, input);
}
