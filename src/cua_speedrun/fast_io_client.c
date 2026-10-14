/* Buffered client for the agent's authenticated loopback HTTP proxy.
 * No TLS/image libraries, redirects, retries, or background process.
 * The proxy owns the persistent remote connection and screenshot files. */
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "Computer request failed: %s; observe before retrying an action.\n", message);
    exit(1);
}

static void timeout_handler(int signum) {
    (void)signum;
    const char message[] = "Computer request timed out; observe before retrying an action.\n";
    ssize_t ignored = write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)ignored;
    _exit(124);
}

static void write_all(int fd, const char *data, size_t length) {
    while (length) {
        ssize_t written = write(fd, data, length);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) fail("write failed");
        data += written;
        length -= (size_t)written;
    }
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGALRM, timeout_handler);
    alarm(120);  /* One deadline covers connect, send, and the entire response. */
    if (argc < 2 || argc > 3) fail("expected an operation and optional action-list JSON");
    int step = !strcmp(argv[1], "step") || !strcmp(argv[1], "step-observe");
    int observe = !strcmp(argv[1], "observe");
    if (!step && !observe && strcmp(argv[1], "done")) fail("unknown operation");
    if (argc != (step ? 3 : 2)) fail("wrong argument count");

    const char prefix[] = "http://127.0.0.1:";
    const char *url = getenv("CS_COMPUTER_URL"), *token = getenv("CS_COMPUTER_TOKEN");
    if (!url || strncmp(url, prefix, sizeof(prefix) - 1)) fail("proxy must be loopback HTTP");
    const char *port_text = url + sizeof(prefix) - 1;
    char *end;
    unsigned long port = strtoul(port_text, &end, 10);
    if (!isdigit((unsigned char)*port_text) || *end || !port || port > 65535) fail("invalid proxy port");
    if (!token || !*token || strlen(token) > 256) fail("invalid proxy token");
    for (const unsigned char *p = (const unsigned char *)token; *p; p++)
        if (!isalnum(*p) && *p != '_' && *p != '-') fail("invalid proxy token");

    char *body = NULL;
    size_t body_length = 0;
    if (step) {
        size_t actions_length = strlen(argv[2]);
        if (actions_length > 16 * 1024 * 1024 - 32) fail("action list too large");
        body = malloc(actions_length + 32);
        if (!body) fail("out of memory");
        int size = snprintf(body, actions_length + 32, "{\"actions\":%s}", argv[2]);
        if (size < 0) fail("could not format action list");
        body_length = (size_t)size;
    }
    char header[1024];
    int header_length = snprintf(header, sizeof(header),
        "%s /%s HTTP/1.1\r\nHost: 127.0.0.1:%lu\r\nX-Gateway-Token: %s\r\n"
        "Accept: application/vnd.cua-speedrun.screenshot-file.v1+json\r\n"
        "Content-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        observe ? "GET" : "POST", argv[1], port, token, body_length);
    if (header_length < 0 || (size_t)header_length >= sizeof(header)) fail("request headers too large");
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    if (fd < 0 || setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one))) fail("socket setup failed");
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons((unsigned short)port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address))) fail("could not connect to proxy");
    write_all(fd, header, (size_t)header_length);
    write_all(fd, body, body_length);
    free(body);

    size_t used = 0, capacity = 8192;
    char *response = malloc(capacity);
    if (!response) fail("out of memory");
    for (;;) {
        if (used + 1 == capacity) {
            if (capacity >= 64 * 1024 * 1024) fail("response exceeds 64 MiB");
            capacity *= 2;
            char *larger = realloc(response, capacity);
            if (!larger) fail("out of memory");
            response = larger;
        }
        ssize_t received = read(fd, response + used, capacity - used - 1);
        if (received < 0 && errno == EINTR) continue;
        if (received < 0) fail("response read failed");
        if (!received) break;
        used += (size_t)received;
    }
    close(fd);
    response[used] = '\0';
    if (used < 12 || (strncmp(response, "HTTP/1.0 ", 9) && strncmp(response, "HTTP/1.1 ", 9)) ||
        !isdigit((unsigned char)response[9]) || !isdigit((unsigned char)response[10]) ||
        !isdigit((unsigned char)response[11])) fail("invalid HTTP status");
    int status = (response[9] - '0') * 100 + (response[10] - '0') * 10 + response[11] - '0';
    char *divider = strstr(response, "\r\n\r\n");
    if (!divider) fail("incomplete HTTP headers");
    unsigned long long content_length = 0;
    int lengths = 0;
    for (char *line = strstr(response, "\r\n") + 2; line < divider; ) {
        char *line_end = strstr(line, "\r\n");
        if (!line_end || line_end > divider) fail("invalid HTTP headers");
        if (!strncasecmp(line, "Transfer-Encoding:", 18)) fail("unexpected transfer encoding");
        if (!strncasecmp(line, "Content-Length:", 15)) {
            const char *value = line + 15;
            while (*value == ' ' || *value == '\t') value++;
            if (!isdigit((unsigned char)*value) || lengths++) fail("invalid Content-Length");
            errno = 0;
            content_length = strtoull(value, &end, 10);
            if (errno || end != line_end) fail("invalid Content-Length");
        }
        line = line_end + 2;
    }
    size_t actual_length = used - (size_t)(divider + 4 - response);
    if (lengths != 1 || content_length != actual_length) fail("truncated response body");
    write_all(STDOUT_FILENO, divider + 4, actual_length);
    write_all(STDOUT_FILENO, "\n", 1);
    free(response);
    alarm(0);
    return status >= 200 && status < 300 ? 0 : 22;
}
