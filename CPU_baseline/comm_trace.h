// common/comm_trace.h — per-exchange network tracing for the 2PC protocol.
//
// Records one CSV row per network exchange (send+recv pair) so the symbolic
// communication model of docs/network_model.md can be calibrated with real
// alpha (per-message latency), beta (effective bandwidth) and skew constants.
//
// Every exchange in this protocol is SYNCHRONOUS and party-serialized
//   party 0:  send; flush; recv        party 1:  recv; send; flush
// so a single wrapper captures the whole cost window; there are no async
// requests whose wait() would have to be timed separately.
//
// Enabled only under -DPCG_ENABLE_PROFILING (the same switch as timing.h);
// otherwise PCG_COMM_TRACE(...) compiles to nothing and no bytes counters,
// clocks or CSV rows exist.
//
// Usage (wrapping an existing exchange):
//     PCG_COMM_TRACE("dpf_gen", level, batch, send_bytes, recv_bytes, {
//         ... the existing send/flush/recv block, verbatim ...
//     });
//
// CSV columns:
//   party,context,phase,level,batch,send_bytes,recv_bytes,
//   t_begin_ns,t_mid_ns,t_end_ns,io_counter_delta
// t_mid_ns is filled by PCG_COMM_TRACE_MID() if the wrapped block wants to
// mark the send/recv boundary; otherwise it equals t_begin_ns.

#ifndef PCG_COMMON_COMM_TRACE_H__
#define PCG_COMMON_COMM_TRACE_H__

#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

namespace pcg_comm {

namespace detail {

inline std::mutex& mtx() { static std::mutex m; return m; }
inline std::ofstream& stream() { static std::ofstream out; return out; }
inline std::string& party() { static std::string p; return p; }
inline std::string& context() { static std::string c; return c; }
inline bool& enabled() { static bool e = false; return e; }

}  // namespace detail

inline void configure(const std::string& path, const std::string& party) {
    std::lock_guard<std::mutex> lock(detail::mtx());
    detail::party() = party;
    detail::enabled() = !path.empty();
    if (!detail::enabled()) return;
    auto& out = detail::stream();
    if (out.is_open()) out.close();
    out.open(path, std::ios::out | std::ios::trunc);
    out << "party,context,phase,level,batch,send_bytes,recv_bytes,"
           "t_begin_ns,t_mid_ns,t_end_ns,io_counter_delta\n";
}

inline void set_context(const std::string& ctx) {
    std::lock_guard<std::mutex> lock(detail::mtx());
    detail::context() = ctx;
}

inline bool active() { return detail::enabled(); }

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline void log(const char* phase, int level, int batch,
                uint64_t send_bytes, uint64_t recv_bytes,
                uint64_t t_begin, uint64_t t_mid, uint64_t t_end,
                uint64_t counter_delta) {
    std::lock_guard<std::mutex> lock(detail::mtx());
    if (!detail::enabled()) return;
    detail::stream() << detail::party() << ',' << detail::context() << ','
                     << phase << ',' << level << ',' << batch << ','
                     << send_bytes << ',' << recv_bytes << ','
                     << t_begin << ',' << t_mid << ',' << t_end << ','
                     << counter_delta << '\n';
}

}  // namespace pcg_comm

#ifdef PCG_ENABLE_PROFILING
// `io` must be an emp::NetIO* (its `counter` field is the sent-byte total).
#define PCG_COMM_TRACE(io, phase, level, batch, send_bytes, recv_bytes, BLOCK) \
    do {                                                                      \
        const uint64_t _ct_c0 = static_cast<uint64_t>((io)->counter);          \
        const uint64_t _ct_t0 = ::pcg_comm::now_ns();                          \
        uint64_t _ct_tm = _ct_t0;                                              \
        BLOCK                                                                  \
        const uint64_t _ct_t1 = ::pcg_comm::now_ns();                          \
        ::pcg_comm::log(phase, (level), (batch),                               \
                        static_cast<uint64_t>(send_bytes),                     \
                        static_cast<uint64_t>(recv_bytes),                     \
                        _ct_t0, _ct_tm, _ct_t1,                                \
                        static_cast<uint64_t>((io)->counter) - _ct_c0);        \
    } while (false)
// Mark the send/recv boundary inside a traced block (optional).
#define PCG_COMM_TRACE_MID() do { _ct_tm = ::pcg_comm::now_ns(); } while (false)
#else
#define PCG_COMM_TRACE(io, phase, level, batch, send_bytes, recv_bytes, BLOCK) \
    do { BLOCK } while (false)
#define PCG_COMM_TRACE_MID() ((void)0)
#endif

#endif  // PCG_COMMON_COMM_TRACE_H__
