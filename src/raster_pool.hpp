#pragma once
// A small fixed thread pool for the native RDP pass: run(count, fn) calls
// fn(0) .. fn(count - 1) spread over the workers and the calling thread, and
// returns once every call has finished. Jobs are handed out through an
// atomic counter, so uneven jobs balance themselves.

#include "common.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class RasterPool {
public:
    // `workers` extra threads (0 = run() executes everything on the caller).
    explicit RasterPool(unsigned workers);
    ~RasterPool();
    RasterPool(const RasterPool&) = delete;
    RasterPool& operator=(const RasterPool&) = delete;

    unsigned workers() const { return static_cast<unsigned>(threads_.size()); }
    void run(u32 count, const std::function<void(u32)>& fn);

private:
    void worker_loop();
    void work();

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    u64 generation_ = 0;  // bumped per run(); workers wait for a new value
    bool quit_ = false;

    // The current job, valid while a run() is in progress.
    const std::function<void(u32)>* fn_ = nullptr;
    u32 count_ = 0;
    std::atomic<u32> next_{0};
    std::atomic<u32> finished_{0};
    unsigned active_ = 0; // workers inside work() for the current run (guarded by mutex_)
};
