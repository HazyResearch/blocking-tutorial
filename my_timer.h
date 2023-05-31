#pragma once
#include <chrono>

class MyTimer {
    std::chrono::time_point<std::chrono::system_clock> start;

public:
    void startCounter() {
        start = std::chrono::system_clock::now();
    }

    int64_t getCounterNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start).count();
    }

    int64_t getCounterMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
    }

    double getCounterMsPrecise() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start).count()
                / 1000000.0;
    }
};




