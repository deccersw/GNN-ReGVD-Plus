/* CWE-416: Use-After-Free */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct Request {
    char method[16];
    char *url;
    int status;
};

struct Request *create_request(const char *method, const char *url) {
    struct Request *req = malloc(sizeof(struct Request));
    strncpy(req->method, method, 15);
    req->method[15] = '\0';
    req->url = strdup(url);
    req->status = 0;
    return req;
}

void free_request(struct Request *req) {
    free(req->url);
    free(req);
}

void process_request(struct Request *req) {
    printf("Processing: %s %s\n", req->method, req->url);
    req->status = 200;
}

int main() {
    struct Request *req = create_request("GET", "/api/data");
    free_request(req);
    process_request(req);  /* use-after-free */
    return 0;
}
