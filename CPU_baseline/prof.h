// common/prof.h — lightweight phase profiler for the PCG-OLE pipeline.
//
// SHIM: minimal reconstruction of the profiling layer the new pcg_ole_impl.h
// expects. Replace with the authoritative common/prof.h from the dev tree
// when it lands (this one preserves the macro/zone API so the impl compiles
// and accumulates per-phase wall time; it does not attempt to match the
// dev tree's exact reporting format).
//
// API used by pcg_ole_impl.h:
//   PCG_PROF_TIC(name)         start a timer named `name`
//   PCG_PROF_TOC(ZONE, name)   stop `name`, add elapsed to accumulator ZONE
//   PCG_PROF_ZONE(ZONE)        RAII scope: whole block counts toward ZONE
//   PCG_PROF_REPORT()          dump accumulated per-zone totals to stderr
//
// Zones (accumulators): PREPROCESSING, DPF_EXPAND, DPF_EXPAND_LOCAL,
//   DPF_EXPAND_COMM, LEAF_CONVERT, NET_WAIT, STEP4.
// Compile with -DPCG_PROF_DISABLE to turn every macro into a no-op.

#ifndef PCG_COMMON_PROF_H__
#define PCG_COMMON_PROF_H__

#include <chrono>
#include <cstdio>

namespace pcg_prof {

enum Zone {
    PREPROCESSING = 0,
    DPF_EXPAND,
    DPF_EXPAND_LOCAL,
    DPF_EXPAND_COMM,
    LEAF_CONVERT,
    NET_WAIT,
    STEP4,
    // Preprocessing broken out (each mixes local compute and its own network
    // rounds; the per-exchange split lives in the comm-trace CSV):
    PRE_KS,        // Kogge-Stone position add  (triples + masks + AND levels)
    PRE_GILBOA,    // payload zp_multiply + zp_triple (3 OT batches)
    PRE_DLTSFT,    // batch_F_DeltaShiftShare circuit (byte-dominant phase)
    DPF_GEN,       // batch_gen_full_eval: levels blocking round trips + local
    NUM_ZONES
};

inline const char* zone_name(int z) {
    static const char* names[NUM_ZONES] = {
        "PREPROCESSING", "DPF_EXPAND", "DPF_EXPAND_LOCAL", "DPF_EXPAND_COMM",
        "LEAF_CONVERT", "NET_WAIT", "STEP4",
        "PRE_KS", "PRE_GILBOA", "PRE_DLTSFT", "DPF_GEN"};
    return (z >= 0 && z < NUM_ZONES) ? names[z] : "?";
}

// Per-thread-agnostic accumulators (single-process-per-party model).
inline double* totals() {
    static double t[NUM_ZONES] = {0};
    return t;
}

using Clock = std::chrono::steady_clock;

inline void add(int zone, double seconds) {
    if (zone >= 0 && zone < NUM_ZONES) totals()[zone] += seconds;
}

struct ScopedZone {
    int zone_;
    Clock::time_point t0_;
    explicit ScopedZone(int z) : zone_(z), t0_(Clock::now()) {}
    ~ScopedZone() {
        add(zone_, std::chrono::duration<double>(Clock::now() - t0_).count());
    }
};

inline void report() {
    std::fprintf(stderr, "== PCG phase profile (s) ==\n");
    for (int z = 0; z < NUM_ZONES; ++z)
        std::fprintf(stderr, "  %-18s %10.4f\n", zone_name(z), totals()[z]);
}

}  // namespace pcg_prof

#ifdef PCG_PROF_DISABLE
  #define PCG_PROF_TIC(name)          ((void)0)
  #define PCG_PROF_TOC(ZONE, name)    ((void)0)
  #define PCG_PROF_ZONE(ZONE)         ((void)0)
  #define PCG_PROF_REPORT()           ((void)0)
#else
  #define PCG_PROF_TIC(name) \
      auto name = ::pcg_prof::Clock::now()
  #define PCG_PROF_TOC(ZONE, name)                                          \
      ::pcg_prof::add(::pcg_prof::ZONE,                                      \
          std::chrono::duration<double>(::pcg_prof::Clock::now() - name).count())
  #define PCG_PROF_ZONE(ZONE) \
      ::pcg_prof::ScopedZone _pcg_zone_##ZONE(::pcg_prof::ZONE)
  #define PCG_PROF_REPORT()           ::pcg_prof::report()
#endif

#endif  // PCG_COMMON_PROF_H__
