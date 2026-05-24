/* CWE-134: Format String Vulnerability — write via %n */
#include <stdio.h>

void log_message(const char *user_msg) {
    printf(user_msg);  /* format string vuln: user controls format */
}

int main() {
    /* %n writes number of bytes printed so far to an address on the stack */
    log_message("%08x.%08x.%08x.%08x.%n");
    return 0;
}
