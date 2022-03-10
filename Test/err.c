#include <stdlib.h>
#include <stdio.h>

void printErr(int flag) {
    switch (flag) {
        case 1:
            fprintf(stderr, "Error: could not allocate memory\n");
            exit(1);
        case 2:
            fprintf(stderr, "Error: No filename found\n");
            exit(2);
    }
}
