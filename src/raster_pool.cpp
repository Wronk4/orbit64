#include "raster_pool.hpp"

RasterPool::RasterPool(unsigned workers) {
    threads_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) threads_.emplace_back([this] { worker_loop(); });
}

RasterPool::~RasterPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : threads_) t.join();
}

void RasterPool::work() {
    for (;;) {
        const u32 i = next_.fetch_add(1, std::memory_order_relaxed);
        if (i >= count_) return;
        (*fn_)(i);
        finished_.fetch_add(1, std::memory_order_release);
    }
}

void RasterPool::worker_loop() {
    u64 seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return quit_ || generation_ != seen; });
            if (quit_) return;
            seen = generation_;
            ++active_;
        }
        work();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_;
        }
        done_.notify_one();
    }
}

void RasterPool::run(u32 count, const std::function<void(u32)>& fn) {
    if (count == 0) return;
    if (threads_.empty() || count == 1) {
        for (u32 i = 0; i < count; ++i) fn(i);
        return;
    }
    {
        // A worker that woke too late for the previous run may still be in
        // work(); it must be out before the job it reads is replaced.
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return active_ == 0; });
        fn_ = &fn;
        count_ = count;
        next_.store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        ++generation_;
    }
    wake_.notify_all();
    work();
    // Every job done, and no worker still between grabbing an index and
    // noticing there is none left (it would read fn_/count_ of the next run).
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return finished_.load(std::memory_order_acquire) == count_ && active_ == 0; });
    fn_ = nullptr;
}
