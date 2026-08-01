#ifndef RAMULATOR_BASE_REQUEST_H
#define RAMULATOR_BASE_REQUEST_H

#include <list>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

#include "base/base.h"

namespace Ramulator {

// Basic request id convention
// 0 = Read, 1 = Write. The device spec defines all others
enum class Type {
    Read = 0,
    Write = 1,
    RefAllBank = 2,
    RefSingleBank = 3,
    AIM = 4,
    MAX = 5
};

enum class MemAccessRegion {
    MIN,
    GPR,
    CFR,
    MEM,
    MAX
};

// ISR Command Opcodes — PCG-specific instruction set (the AiM GEMV ISA: MAC/AF/GB/GPR/EWMUL
// has been removed; only the DPF/GGM compute op + control sentinels remain).
// NOTE: m_aim_requests in the DRAM spec is indexed by (int)opcode, so its order must match these.
enum class Opcode {
    MIN = 0,
    ISR_GGM_EXTEND = 1,  // DPF/GGM ChaCha tree expansion: read [opsize] parent seed column-words
                         // from [row_addr=row_in] of [bank_index], compute ChaCha8 ([compute_latency]
                         // FPU cycles), write [2*opsize] child column-words to [row_out]/[write_bank].
    ISR_GGM_REDUCE = 2,  // DPF leaf reduction (the OTHER half of the complete DPF): read [opsize] leaf
                         // column-words from [bank_index]/[row_addr]/[col_in], apply the per-leaf
                         // 62-bit modular multiply ([compute_latency]) for the CW-scale, and write
                         // [wrsize] result column-words to [write_bank]/[row_out]/[col_out] (the
                         // scatter into g, or the 2N->N fold). Read:write ratio is configurable
                         // (leaves dominate, g is t-times smaller), unlike EXTEND's fixed 1:2.
    ISR_EOC = 3,         // End of compute for the current kernel (mandatory trace sentinel).
    ISR_SYNC = 4,        // Barrier (control).
    ISR_NET_DELAY = 5,   // Frontend-only network timeline: advance the single NIC-lane scalar by
                         // [opsize] rungs x [compute_latency] CK (one rung = one party-serialized
                         // round trip). Non-blocking; never reaches a controller.
    ISR_NET_WAIT = 6,    // Frontend-only network barrier: hold the dispatch queue head until
                         // m_clk >= the NIC-lane scalar (max() semantics for the co-schedule).
    MAX = 7
};

struct Request {
    Addr_t addr = -1;
    Data_t data;
    AddrVec_t addr_vec{};
    int host_req_id = -1;
    int AiM_req_id = -1;

    Type type = Type::MAX;

    MemAccessRegion mem_access_region = MemAccessRegion::MAX;

    Opcode opcode = Opcode::MAX;

    int32_t opsize = -1;

    // This request will be broadcasted/multicasted to the channels
    // whose bit is set in channel mask.
    // NOT USED in ISR_EWADD.
    // Channel mask must show 1 channel in ISR_WR_ABK ISR
    int64_t channel_mask = -1;

    // Target bank (ISR_GGM_EXTEND: the parent/read bank; the store bank is write_bank).
    int16_t bank_index = -1;

    // Bank row and column address (ISR_GGM_EXTEND: row_addr = row_in, the parent row).
    int32_t row_addr = -1;
    int32_t col_addr = -1;

    // ISR_GGM_EXTEND: destination row for the child writes. row_addr is the parent (source) row.
    int32_t row_out = -1;

    // ISR_GGM_EXTEND: ChaCha8 compute latency in FPU cycles. Charged once per op, overlapping the
    // memory accesses (see AiMDRAMController). -1 (default) means no compute charge (memory-only).
    int32_t compute_latency = -1;

    // Pseudochannel index (HBM2-style hierarchies). -1 (default) maps to 0 in apply_addr_mapp.
    int16_t pseudochannel = -1;

    // ISR_GGM_EXTEND: starting column for the read (col_in) and write (col_out) phases, so multiple
    // batches can be packed into one row (row-buffer locality). Reads use col_in..col_in+opsize-1,
    // writes use col_out..col_out+2*opsize-1. Default 0.
    int32_t col_in = -1;
    int32_t col_out = -1;

    // ISR_GGM_REDUCE: number of result (g) column-words written. EXTEND derives writes = 2*opsize,
    // but REDUCE's read:write ratio differs (leaves >> g), so it is given explicitly. -1 => 0.
    int32_t wrsize = -1;

    // ISR_GGM_EXTEND double-buffering: bank for the store phase. -1 (default) => same as bank_index
    // (children written to the parent bank, the row-conflicting baseline). Set to the PIM unit's
    // PARTNER bank (read A / write B) to keep load and store row buffers from evicting each other.
    int16_t write_bank = -1;

    // ALL-BANK BROADCAST (faithful AiM / HBM-PIM SIMD): one ISR drives `nbanks` banks of the channel
    // in lockstep — each bank runs the same op (same row/col offsets) on its own data, in parallel.
    // The dispatcher issues ONE fat op instead of nbanks single-bank ops (the real PIM behavior), so
    // bank-level parallelism is not bottlenecked by the serial host issue port. -1/1 => single bank.
    int16_t nbanks = 1;

    // DRU / SM sequential column stream (host GGM_REDUCE with compute_latency<=0):
    // consumed strictly in order by the channel-level DRU, needs no FRFCFS
    // reordering. Routed to the controller's dedicated O(1) FIFO path so the
    // stream cannot inflate the scanned read/write buffers (the simulator's
    // per-cycle get_best_request scan is O(buffer)).
    bool is_dru = false;

    int source_id = -1; // An identifier for where the request is coming from (e.g., which core)

    int command = -1;       // The command that need to be issued to progress the request
    int final_command = -1; // The final command that is needed to finish the request

    Clk_t arrive = -1; // Clock cycle when the request arrive at the memory controller
    Clk_t issue = -1;  // Clock cycle when the request issued at the memory controller
    Clk_t depart = -1; // Clock cycle when the request depart the memory controller

    int type_id = -1;

    std::function<void(Request &)> callback;

    Request(Addr_t addr, int type_id);
    Request(AddrVec_t addr_vec, int type_id);
    Request(Addr_t addr, int type_id, int source_id, std::function<void(Request &)> callback);
    // Request(Request const &req);

    std::string str();

    bool is_reader();
};

class AiMISR {

public:
    enum class Field {
        opsize,
        channel_mask,
        bank_index,
        row_addr,
        row_out,
        compute_latency,
        col_in,
        col_out,
        write_bank,
        wrsize,
        nbanks
    };

    std::map<Field, std::string> field_to_str = {
        {Field::opsize, "opsize"},
        {Field::channel_mask, "channel_mask"},
        {Field::bank_index, "bank_index"},
        {Field::row_addr, "row_addr"},
        {Field::row_out, "row_out"},
        {Field::compute_latency, "compute_latency"},
        {Field::col_in, "col_in"},
        {Field::col_out, "col_out"},
        {Field::write_bank, "write_bank"},
        {Field::wrsize, "wrsize"},
        {Field::nbanks, "nbanks"},
    };

    AiMISR(Opcode opcode_ = Opcode::MAX,
           std::vector<Field> legal_fields_ = {},
           bool channel_count_eq_one_ = false,
           bool AiM_DMA_blocking_ = false,
           bool require_reg_RW_mod_ = false,
           std::string target_level_ = "")
        : opcode(opcode_),
          legal_fields(legal_fields_),
          channel_count_eq_one(channel_count_eq_one_),
          AiM_DMA_blocking(AiM_DMA_blocking_),
          require_reg_RW_mod(require_reg_RW_mod_),
          target_level(target_level_) {}

    Opcode opcode;

    std::vector<Field> legal_fields;

    bool channel_count_eq_one;
    bool AiM_DMA_blocking;
    bool require_reg_RW_mod;

    std::string target_level;

    bool is_field_legal(Field field) {
        if (std::count(legal_fields.begin(), legal_fields.end(), field))
            return true;
        return false;
    }

    template <typename T>
    void is_field_value_legal(Field field, T value);
};

class AiMISRInfo {
private:
    static std::map<std::string, AiMISR> opcode_str_to_aim_ISR;
    static std::map<Opcode, std::string> aim_opcode_to_str;

    static std::map<std::string, Type> str_to_type;
    static std::map<Type, std::string> type_to_str;

    static std::map<std::string, MemAccessRegion> str_to_mem_access_region;
    static std::map<MemAccessRegion, std::string> mem_access_region_to_str;

public:
    static void init() {
        aim_opcode_to_str[Opcode::ISR_SYNC] = "ISR_SYNC";
        opcode_str_to_aim_ISR["ISR_SYNC"] = AiMISR(Opcode::ISR_SYNC,
                                                   {},
                                                   false,    // channel_count_eq_one
                                                   true,     // AiM_DMA_blocking
                                                   false,    // require_reg_RW_mod
                                                   "channel" // target_level
        );

        aim_opcode_to_str[Opcode::ISR_EOC] = "ISR_EOC";
        opcode_str_to_aim_ISR["ISR_EOC"] = AiMISR(Opcode::ISR_EOC,
                                                  {},
                                                  false, // channel_count_eq_one
                                                  true,  // AiM_DMA_blocking
                                                  false, // require_reg_RW_mod
                                                  "DMA"  // target_level
        );

        aim_opcode_to_str[Opcode::ISR_NET_DELAY] = "ISR_NET_DELAY";
        opcode_str_to_aim_ISR["ISR_NET_DELAY"] = AiMISR(Opcode::ISR_NET_DELAY,
                                                        {AiMISR::Field::opsize,
                                                         AiMISR::Field::compute_latency},
                                                        false, // channel_count_eq_one
                                                        false, // AiM_DMA_blocking (frontend-only)
                                                        false, // require_reg_RW_mod
                                                        "DMA"  // target_level (never dispatched)
        );

        aim_opcode_to_str[Opcode::ISR_NET_WAIT] = "ISR_NET_WAIT";
        opcode_str_to_aim_ISR["ISR_NET_WAIT"] = AiMISR(Opcode::ISR_NET_WAIT,
                                                       {},
                                                       false, // channel_count_eq_one
                                                       false, // AiM_DMA_blocking (frontend-only)
                                                       false, // require_reg_RW_mod
                                                       "DMA"  // target_level (never dispatched)
        );

        aim_opcode_to_str[Opcode::ISR_GGM_EXTEND] = "ISR_GGM_EXTEND";
        opcode_str_to_aim_ISR["ISR_GGM_EXTEND"] = AiMISR(Opcode::ISR_GGM_EXTEND,
                                                         {AiMISR::Field::opsize,
                                                          AiMISR::Field::channel_mask,
                                                          AiMISR::Field::bank_index,
                                                          AiMISR::Field::row_addr, // = row_in
                                                          AiMISR::Field::row_out,
                                                          AiMISR::Field::compute_latency,
                                                          AiMISR::Field::col_in,
                                                          AiMISR::Field::col_out,
                                                          AiMISR::Field::write_bank,
                                                          AiMISR::Field::nbanks},  // all-bank broadcast
                                                         false,   // channel_count_eq_one
                                                         false,   // AiM_DMA_blocking
                                                         false,   // require_reg_RW_mod
                                                         "column" // target_level
        );

        aim_opcode_to_str[Opcode::ISR_GGM_REDUCE] = "ISR_GGM_REDUCE";
        opcode_str_to_aim_ISR["ISR_GGM_REDUCE"] = AiMISR(Opcode::ISR_GGM_REDUCE,
                                                         {AiMISR::Field::opsize,         // leaf reads
                                                          AiMISR::Field::channel_mask,
                                                          AiMISR::Field::bank_index,     // leaf bank
                                                          AiMISR::Field::row_addr,       // leaf row
                                                          AiMISR::Field::row_out,        // g row
                                                          AiMISR::Field::compute_latency,// modmul
                                                          AiMISR::Field::col_in,         // leaf col
                                                          AiMISR::Field::col_out,        // g col
                                                          AiMISR::Field::write_bank,     // g bank
                                                          AiMISR::Field::wrsize,         // g writes
                                                          AiMISR::Field::nbanks},        // all-bank broadcast
                                                         false,   // channel_count_eq_one
                                                         false,   // AiM_DMA_blocking
                                                         false,   // require_reg_RW_mod
                                                         "column" // target_level
        );

        str_to_type["R"] = Type::Read;
        type_to_str[Type::Read] = "R";
        str_to_type["W"] = Type::Write;
        type_to_str[Type::Write] = "W";
        str_to_type["AiM"] = Type::AIM;
        type_to_str[Type::AIM] = "AiM";

        str_to_mem_access_region["GPR"] = MemAccessRegion::GPR;
        mem_access_region_to_str[MemAccessRegion::GPR] = "GPR";
        str_to_mem_access_region["CFR"] = MemAccessRegion::CFR;
        mem_access_region_to_str[MemAccessRegion::CFR] = "CFR";
        str_to_mem_access_region["MEM"] = MemAccessRegion::MEM;
        mem_access_region_to_str[MemAccessRegion::MEM] = "MEM";
    }

    static bool type_valid(std::string type_str) {
        if (str_to_type.size() == 0) {
            throw ConfigurationError("AiMISRInfo not initialized!");
        }

        if (str_to_type.find(type_str) == str_to_type.end())
            return false;
        return true;
    }

    static bool type_valid(Type type) {
        if (type_to_str.size() == 0) {
            throw ConfigurationError("AiMISRInfo not initialized!");
        }

        if (type_to_str.find(type) == type_to_str.end())
            return false;
        return true;
    }

    static Type convert_str_to_type(std::string type_str) {
        if (type_valid(type_str) == false) {
            throw ConfigurationError("Trace: unknown type {}!", type_str.c_str());
        }
        return str_to_type[type_str];
    }

    static std::string convert_type_to_str(Type type) {
        if (type_valid(type) == false) {
            throw ConfigurationError("Trace: unknown type {}!", (int)type);
        }
        return type_to_str[type];
    }

    static bool AiM_opcode_valid(std::string AiM_opcode_str) {
        if (opcode_str_to_aim_ISR.size() == 0) {
            throw ConfigurationError("AiMISRInfo not initialized!");
        }

        if (opcode_str_to_aim_ISR.find(AiM_opcode_str) == opcode_str_to_aim_ISR.end())
            return false;
        return true;
    }

    static bool AiM_opcode_valid(Opcode AiM_opcode) {
        if (aim_opcode_to_str.size() == 0) {
            throw ConfigurationError("AiMISRInfo not initialized!");
        }

        if (aim_opcode_to_str.find(AiM_opcode) == aim_opcode_to_str.end())
            return false;
        return true;
    }

    static AiMISR convert_opcode_str_to_AiM_ISR(std::string AiM_opcode_str) {
        if (AiM_opcode_valid(AiM_opcode_str) == false) {
            throw ConfigurationError("Trace: unknown AiM opcode {}!", AiM_opcode_str.c_str());
        }
        return opcode_str_to_aim_ISR[AiM_opcode_str];
    }

    static AiMISR convert_AiM_opcode_to_AiM_ISR(Opcode AiM_opcode) {
        return opcode_str_to_aim_ISR[convert_AiM_opcode_to_str(AiM_opcode)];
    }

    static std::string convert_AiM_opcode_to_str(Opcode AiM_opcode) {
        if (AiM_opcode_valid(AiM_opcode) == false) {
            throw ConfigurationError("Trace: unknown AiM opcode {}!", (int)AiM_opcode);
        }
        return aim_opcode_to_str[AiM_opcode];
    }

    static bool mem_access_region_valid(std::string mem_access_region_str) {
        if (str_to_mem_access_region.size() == 0) {
            throw ConfigurationError("AiMISRInfo not initialized!");
        }

        if (str_to_mem_access_region.find(mem_access_region_str) == str_to_mem_access_region.end())
            return false;
        return true;
    }

    static bool mem_access_region_valid(MemAccessRegion mem_access_region) {
        if (mem_access_region_to_str.size() == 0) {
            throw ConfigurationError("AiMISRInfo not initialized!");
        }

        if (mem_access_region_to_str.find(mem_access_region) == mem_access_region_to_str.end())
            return false;
        return true;
    }

    static MemAccessRegion convert_str_to_mem_access_region(std::string mem_access_region_str) {
        if (mem_access_region_valid(mem_access_region_str) == false) {
            throw ConfigurationError("Trace: unknown mem_access_region {}!", mem_access_region_str.c_str());
        }
        return str_to_mem_access_region[mem_access_region_str];
    }

    static std::string convert_mem_access_region_to_str(MemAccessRegion mem_access_region) {
        if (mem_access_region_valid(mem_access_region) == false) {
            throw ConfigurationError("Trace: unknown mem_access_region {}!", (int)mem_access_region);
        }
        return mem_access_region_to_str[mem_access_region];
    }

    static bool opcode_requires_reg_RW_mod(Opcode AiM_opcode) {
        return convert_AiM_opcode_to_AiM_ISR(AiM_opcode).require_reg_RW_mod;
    }
};

static AiMISRInfo AiM_host_request_info();

struct ReqBuffer {
    std::list<Request> buffer;
    size_t max_size = 32;

    using iterator = std::list<Request>::iterator;
    iterator begin() { return buffer.begin(); };
    iterator end() { return buffer.end(); };

    size_t size() const { return buffer.size(); }

    bool enqueue(const Request &request) {
        if (buffer.size() <= max_size) {
            buffer.push_back(request);
            return true;
        } else {
            return false;
        }
    }

    void remove(iterator it) {
        buffer.erase(it);
    }
};

template <typename T>
void AiMISR::is_field_value_legal(AiMISR::Field field, T value) {
    if (is_field_legal(field)) {
        if (value == (T)-1) {
            printf("Trace: opcode %s must be provided with field %s!", AiMISRInfo::convert_AiM_opcode_to_str(opcode).c_str(), field_to_str[field].c_str());
            exit(-1);
        }
    } else {
        if (value != (T)-1) {
            printf("Trace: opcode %s does not accept field %s!", AiMISRInfo::convert_AiM_opcode_to_str(opcode).c_str(), field_to_str[field].c_str());
            exit(-1);
        }
    }
}

} // namespace Ramulator

#endif // RAMULATOR_BASE_REQUEST_H