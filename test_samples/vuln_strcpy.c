/* CWE-121: Stack Buffer Overflow via strcpy */
#include <string.h>
#include <stdio.h>

void process_input(const char *user_data) {
    char local_buf[16];
    strcpy(local_buf, user_data);
    printf("Processed: %s\n", local_buf);
}

int main() {
    char payload[256];
    memset(payload, 'A', 255);
    payload[255] = '\0';
    process_input(payload);
    return 0;
}
