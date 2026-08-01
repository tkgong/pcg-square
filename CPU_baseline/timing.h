#ifndef COMMON_TIMING_H__
#define COMMON_TIMING_H__

#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>

namespace pcg_profile {

namespace detail {

inline std::mutex& profile_mutex() {
    static std::mutex m;
    return m;
}

inline std::ofstream& profile_stream() {
    static std::ofstream out;
    return out;
}

inline std::string& profile_party() {
    static std::string party;
    return party;
}

inline std::string& profile_context() {
    static std::string context;
    return context;
}

inline bool& profile_enabled() {
    static bool enabled = false;
    return enabled;
}

inline std::string csv_escape(const std::string& in) {
    bool quote = false;
    for (char c : in) {
        if (c == ',' || c == '"' || c == '\n') quote = true;
    }
    if (!quote) return in;
    std::string out = "\"";
    for (char c : in) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

}  // namespace detail

inline void configure(const std::string& path, const std::string& party) {
    std::lock_guard<std::mutex> lock(detail::profile_mutex());
    detail::profile_party() = party;
    detail::profile_enabled() = !path.empty();
    if (!detail::profile_enabled()) return;

    auto& out = detail::profile_stream();
    if (out.is_open()) out.close();
    out.open(path, std::ios::out | std::ios::trunc);
    out << "party,context,phase,ms\n";
}

inline void set_context(const std::string& context) {
    std::lock_guard<std::mutex> lock(detail::profile_mutex());
    detail::profile_context() = context;
}

inline void record(const std::string& phase, double ms) {
    std::lock_guard<std::mutex> lock(detail::profile_mutex());
    if (!detail::profile_enabled()) return;
    auto& out = detail::profile_stream();
    out << detail::csv_escape(detail::profile_party()) << ','
        << detail::csv_escape(detail::profile_context()) << ','
        << detail::csv_escape(phase) << ','
        << ms << '\n';
    out.flush();
}

class ScopedTimer {
public:
    explicit ScopedTimer(std::string phase)
        : phase_(std::move(phase)),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto stop = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(stop - start_).count();
        record(phase_, ms);
    }

private:
    std::string phase_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace pcg_profile

#define PCG_PROFILE_CONCAT_IMPL(x, y) x##y
#define PCG_PROFILE_CONCAT(x, y) PCG_PROFILE_CONCAT_IMPL(x, y)
#ifdef PCG_ENABLE_PROFILING
#define PCG_PROFILE_SCOPE(name) \
    pcg_profile::ScopedTimer PCG_PROFILE_CONCAT(_pcg_profile_scope_, __LINE__)(name)
#else
#define PCG_PROFILE_SCOPE(name) do { (void)sizeof(name); } while (false)
#endif

#endif  // COMMON_TIMING_H__
