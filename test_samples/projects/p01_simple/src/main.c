#include "util.h"
#include <stdio.h>
#include <string.h>

static void log_line(const char *msg) {
    fprintf(stderr, "[log] %s\n", msg);
}

void handle_request(const char *user_input) {
    char buf[64];
    copy_data(buf, user_input);
    int c = checksum(buf, strlen(buf));
    printf("%d\n", c);
    log_line(buf);
}
