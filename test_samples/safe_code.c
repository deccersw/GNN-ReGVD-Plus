/* Safe code — strncpy with bounds check */
#include <string.h>
#include <stdio.h>

void process_input(const char *user_data) {
    char local_buf[16];
    strncpy(local_buf, user_data, sizeof(local_buf) - 1);
    local_buf[sizeof(local_buf) - 1] = '\0';
    printf("Processed: %s\n", local_buf);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        process_input(argv[1]);
    }
    return 0;
}
