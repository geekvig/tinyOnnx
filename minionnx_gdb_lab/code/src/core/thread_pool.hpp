// =====================================================================================
// CORE · thread_pool.hpp  --  a small thread pool with submit() and parallel_for()
// =====================================================================================
// Picture a kitchen. The pool's worker threads are cooks waiting for order tickets.
// submit(task) pins a ticket on the board; a free cook takes it and cooks it.
// parallel_for(0, N, grain, fn) cuts one big job ("chop 1000 onions") into chunks of
// `grain` and lets every cook -- AND the person who asked (the calling thread) -- grab
// chunks until none are left.
//
// Design rules (each one fixed a real problem, see the PDF chapter "Threads"):
//   1. "N threads" = N-1 workers + the calling thread.   N == 1 -> no workers, all inline.
//   2. The CALLER also works in parallel_for, claiming chunks with an atomic counter. So a
//      parallel_for called from inside a worker (nested parallelism) can always finish on
//      its own: it can never deadlock waiting for a busy pool.
//   3. Chunk boundaries depend only on `grain`, never on the number of threads, so results
//      are bit-identical for any thread count.
//   4. Workers spin ~200us before sleeping; waking a sleeping thread costs ~1ms on some VMs.
//   5. NEW in MiniOnnx: an exception thrown inside fn on ANY thread is caught, the remaining
//      chunks are skipped, and the first exception is rethrown on the caller. (In the
//      original, a throw on a worker thread called std::terminate and killed the process.)
// =====================================================================================
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mo {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    __builtin_ia32_pause();                 // x86 PAUSE: "I'm spinning", saves power, helps hyperthreads
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");     // ARM equivalent
#else
    std::this_thread::yield();
#endif
}

class ThreadPool {
public:
    explicit ThreadPool(int logical_threads) {
        n_ = std::max(1, logical_threads);
        for (int i = 0; i < n_ - 1; i++) workers_.emplace_back([this] { worker_loop(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);    // set the flag UNDER the lock: no lost wake-up
            stopping_.store(true);
        }
        cv_.notify_all();
        for (auto &t : workers_) t.join();          // a pool never outlives its threads
    }
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int num_threads() const { return n_; }
    int num_workers() const { return int(workers_.size()); }

    void submit(std::function<void()> task) {
        if (workers_.empty()) { task(); return; }   // inline mode
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push(std::move(task));
            qsize_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_one();
    }

    // Calls fn(chunk_begin, chunk_end) for each `grain`-sized chunk of [begin, end).
    // Returns when every chunk is done. Rethrows the first exception thrown by fn.
    template <class F>
    void parallel_for(int64_t begin, int64_t end, int64_t grain, F &&fn) {
        const int64_t n = end - begin;
        if (n <= 0) return;
        grain = std::max<int64_t>(1, grain);
        const int64_t nchunks = (n + grain - 1) / grain;
        if (nchunks <= 1 || workers_.empty()) {      // nothing to split, or no helpers: just do it
            for (int64_t b = begin; b < end; b += grain) fn(b, std::min(end, b + grain));
            return;
        }

        struct State {
            std::atomic<int64_t> next{0}, done{0};
            std::atomic<bool> failed{false};
            std::exception_ptr error;               // written once, by the thread that set `failed`
            std::mutex m;
            std::condition_variable cv;
        };
        auto st = std::make_shared<State>();
        auto *fnp = &fn;
        // A helper that starts after we return finds no chunk to claim and never touches fn.
        auto run = [st, begin, end, grain, nchunks, fnp]() {
            for (;;) {
                const int64_t c = st->next.fetch_add(1);
                if (c >= nchunks) return;
                if (!st->failed.load(std::memory_order_relaxed)) {
                    const int64_t b = begin + c * grain, e = std::min(end, b + grain);
                    try { (*fnp)(b, e); }
                    catch (...) {
                        if (!st->failed.exchange(true)) st->error = std::current_exception();
                    }
                }
                if (st->done.fetch_add(1, std::memory_order_acq_rel) + 1 == nchunks) {
                    std::lock_guard<std::mutex> lk(st->m);
                    st->cv.notify_all();
                }
            }
        };
        const int64_t helpers = std::min<int64_t>(num_workers(), nchunks - 1);
        for (int64_t i = 0; i < helpers; i++) submit(run);
        run();                                                       // the caller works too (rule 2)
        auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        while (st->done.load(std::memory_order_acquire) != nchunks && std::chrono::steady_clock::now() < deadline)
            cpu_relax();
        if (st->done.load(std::memory_order_acquire) != nchunks) {
            std::unique_lock<std::mutex> lk(st->m);
            st->cv.wait(lk, [&] { return st->done.load(std::memory_order_acquire) == nchunks; });
        }
        // MOVE the error out: then only this thread holds the exception object, and a helper
        // that destroys State later never touches it (keeps ThreadSanitizer quiet & correct).
        if (std::exception_ptr e = std::move(st->error)) std::rethrow_exception(e);
    }

private:
    void worker_loop() {
        using clock = std::chrono::steady_clock;
        for (;;) {
            auto deadline = clock::now() + std::chrono::microseconds(200);
            while (qsize_.load(std::memory_order_acquire) == 0 && !stopping_.load() && clock::now() < deadline)
                cpu_relax();
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stopping_.load() || !queue_.empty(); });
                if (queue_.empty()) return;                 // stopping and nothing left to do
                task = std::move(queue_.front());
                queue_.pop();
                qsize_.fetch_sub(1, std::memory_order_release);
            }
            task();
        }
    }

    int n_ = 1;
    std::atomic<bool> stopping_{false};
    std::atomic<int> qsize_{0};
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
};

} // namespace mo
