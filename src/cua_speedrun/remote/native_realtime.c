/* Keep the controller's UTC independent of the desktop's CLOCK_REALTIME.
 * Only wall-clock reads are translated; elapsed-time clocks stay unchanged.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static int64_t utc_offset_ns;
static int ready;

__attribute__((constructor)) static void initialize_clock(void) {
    const char *value = getenv("CS_NATIVE_UTC_OFFSET_NS");
    char *end = NULL;
    errno = 0;
    if (value != NULL) {
        utc_offset_ns = strtoll(value, &end, 10);
    }
    if (value == NULL || end == value || *end != '\0' || errno != 0) {
        static const char message[] = "Invalid native controller UTC anchor\n";
        write(STDERR_FILENO, message, sizeof(message) - 1);
        _exit(127);
    }
    ready = 1;
}

int cua_native_realtime_active(void) {
    return ready;
}

static int controller_time(struct timespec *result) {
    struct timespec monotonic;
    if (syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &monotonic) != 0) {
        return -1;
    }
    int64_t now = utc_offset_ns + (int64_t)monotonic.tv_sec * 1000000000
                  + monotonic.tv_nsec;
    result->tv_sec = now / 1000000000;
    result->tv_nsec = now % 1000000000;
    if (result->tv_nsec < 0) {
        result->tv_sec--;
        result->tv_nsec += 1000000000;
    }
    return 0;
}

int clock_gettime(clockid_t clock, struct timespec *result) {
    if (ready && (clock == CLOCK_REALTIME || clock == CLOCK_REALTIME_COARSE)) {
        return controller_time(result);
    }
    return syscall(SYS_clock_gettime, clock, result);
}

int gettimeofday(struct timeval *result, void *timezone) {
    if (!ready) {
        return syscall(SYS_gettimeofday, result, timezone);
    }
    struct timespec now;
    if (controller_time(&now) != 0) {
        return -1;
    }
    result->tv_sec = now.tv_sec;
    result->tv_usec = now.tv_nsec / 1000;
    if (timezone != NULL) {
        return syscall(SYS_gettimeofday, NULL, timezone);
    }
    return 0;
}

time_t time(time_t *result) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
        return (time_t)-1;
    }
    if (result != NULL) {
        *result = now.tv_sec;
    }
    return now.tv_sec;
}
