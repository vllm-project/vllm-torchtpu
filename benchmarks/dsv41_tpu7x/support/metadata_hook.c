#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

// Function pointer to original connect
int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }

    if (addr->sa_family == AF_INET) {
        struct sockaddr_in *addr_in = (struct sockaddr_in *)addr;
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr_in->sin_addr), ip, INET_ADDRSTRLEN);
        
        // Check if it is the metadata server IP
        if (strcmp(ip, "169.254.169.254") == 0) {
            // Redirect to 127.0.0.1:1 (dead port) to force failure
            char comm[16] = "unknown";
            int fd = open("/proc/self/comm", O_RDONLY);
            if (fd >= 0) {
                int n = read(fd, comm, sizeof(comm) - 1);
                if (n > 0) {
                    comm[n] = '\0';
                    // Strip newline if present
                    if (comm[n-1] == '\n') comm[n-1] = '\0';
                }
                close(fd);
            }
            
            // Get current time with milliseconds
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            struct tm *tm_info = localtime(&ts.tv_sec);
            char time_str[32];
            strftime(time_str, sizeof(time_str), "%H:%M:%S", tm_info);
            
            char msg[256];
            int len = snprintf(msg, sizeof(msg), "[LD_PRELOAD] [%s] [%s:%d] Redirecting connect to %s:%d to 127.0.0.1:80\n", 
                               time_str, comm, getpid(), ip, ntohs(addr_in->sin_port));
            if (len > 0) {
                (void)write(2, msg, len); // Write to stderr
            }
            addr_in->sin_addr.s_addr = inet_addr("127.0.0.1");
            addr_in->sin_port = htons(80); // Mock metadata port
        }
    }

    return real_connect(sockfd, addr, addrlen);
}
