#include "addr_mapper/addr_mapper.h"
#include "base/request.h"
#include "base/type.h"
#include "dram/dram.h"
#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <math.h>
#include <vector>

namespace Ramulator {

#define ISR_SIZE (1 << 21)
#define MAX_CHANNEL_COUNT 1024
// Banks per channel for the GGM all-bank broadcast (HBM2_4Gb: 4 bankgroups x 4 banks = 16).
// Lane bk maps to bank (base + 2*bk) % NUM_GGM_BANKS, matching the Python NUM_BANKS in ggm_addr.py.
#define NUM_GGM_BANKS 16

class AiMDRAMSystem final : public IMemorySystem, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, AiMDRAMSystem, "AiMDRAM", "AiM memory system (AiM DMA).");

protected:
    Clk_t m_clk = 0;
    IDRAM *m_dram;
    int m_num_levels = -1;
    bool m_has_rank = false; // Does the DRAM have rank level?
    IAddrMapper *m_addr_mapper;
    std::vector<IDRAMController *> m_controllers;
    Logger_t m_logger;
    std::queue<Request> request_queue;
    std::queue<Request> remaining_AiM_requests[MAX_CHANNEL_COUNT];
    int AiM_req_id = 0;

    int stalled_AiM_requests = 0;

    // Frontend-only network timeline (ISR_NET_DELAY / ISR_NET_WAIT): a single
    // NIC-lane scalar. DELAY advances it by rungs*alpha (rungs serialize by
    // construction); WAIT holds the dispatch head until m_clk catches up.
    Clk_t m_net_ready_at = 0;
    Clk_t s_net_busy_cycles = 0;   // total NIC-lane occupancy (for U_NET)

    std::function<void(Request &)> callback;

    uint8_t CountSetBit(const int64_t ch_mask) const {
        assert(ch_mask > 0);

        uint8_t count = 0;

        for (int i = 0; i < MAX_CHANNEL_COUNT; i++)
            if (ch_mask & (0x1 << i))
                count++;

        return count;
    }

    uint8_t FindFirstChannelIndex(int64_t &ch_mask) const {
        uint32_t ch_mask_u = ch_mask;
        assert(ch_mask_u & 0xffffffff != 0);

        for (int i = 0; i < MAX_CHANNEL_COUNT; i++) {
            if (ch_mask_u & (0x1 << i)) {
                ch_mask_u &= ~(0x1 << i);
                ch_mask = ch_mask_u;
                return MAX_CHANNEL_COUNT - 1 - i;
            }
        }

        assert(false);
        return 0;
    }

    void apply_addr_mapp(Request &req, int channel_id) {
        req.addr_vec.resize(m_num_levels, -1);
        if ((channel_id < 0) || (channel_id >= MAX_CHANNEL_COUNT)) {
            m_logger->error("{} has CH more than {}!", req.str(), MAX_CHANNEL_COUNT);
            exit(-1);
        }
        req.addr_vec[m_dram->m_levels("channel")] = channel_id;
        // Set any intermediate level(s) between channel and bankgroup (rank for GDDR-like specs,
        // pseudochannel for HBM2) to the requested value (default 0). Generic so the same code
        // works whether the hierarchy has rank, pseudochannel, or neither.
        int ch_idx = m_dram->m_levels("channel");
        int bg_idx = m_dram->m_levels("bankgroup");
        for (int lvl = ch_idx + 1; lvl < bg_idx; lvl++) {
            req.addr_vec[lvl] = (req.pseudochannel >= 0) ? req.pseudochannel : 0;
        }
        if (req.bank_index == -1) {
            req.addr_vec[m_dram->m_levels("bankgroup")] = -1;
            req.addr_vec[m_dram->m_levels("bank")] = -1;
        } else {
            int banks_per_bg = m_dram->get_level_size("bank");
            int num_bg = m_dram->get_level_size("bankgroup");
            if ((req.bank_index < 0) || (req.bank_index >= banks_per_bg * num_bg)) {
                m_logger->error("{} has BA more than {}!", req.str(), banks_per_bg * num_bg);
                exit(-1);
            }
            req.addr_vec[m_dram->m_levels("bankgroup")] = req.bank_index / banks_per_bg;
            req.addr_vec[m_dram->m_levels("bank")] = req.bank_index % banks_per_bg;
        }
        req.addr_vec[m_dram->m_levels("row")] = req.row_addr;
        req.addr_vec[m_dram->m_levels("column")] = req.col_addr;
    }

public:
    std::map<Type, std::map<MemAccessRegion, int>> s_num_RW_requests;
    std::map<Opcode, int> s_num_AiM_requests;
    int s_ISR_queue_full = 0;
    int s_wait_RD_stall = 0;

public:
    void init() override {
        // Create device (a top-level node wrapping all channel nodes)
        m_dram = create_child_ifce<IDRAM>();
        m_num_levels = m_dram->m_levels.size();
        m_addr_mapper = create_child_ifce<IAddrMapper>();

        m_logger = Logging::create_logger("AiMDRAMSystem");

        if (m_dram->m_levels("bankgroup") - m_dram->m_levels("channel") == 1) {
            m_has_rank = false;
            m_logger->info("AiMDRAMSystem: No intermediate level between channel and bankgroup.");
        } else if (m_dram->m_levels("bankgroup") - m_dram->m_levels("channel") == 2) {
            m_has_rank = true;
            m_logger->info("AiMDRAMSystem: One intermediate level (rank/pseudochannel) between channel and bankgroup.");
        } else {
            throw ConfigurationError("AiMDRAMSystem: Invalid number of levels in DRAM {}!", m_dram->get_name());
        }

        int num_channels = m_dram->get_level_size("channel");

        // Create memory controllers
        for (int i = 0; i < num_channels; i++) {
            IDRAMController *controller = create_child_ifce<IDRAMController>();
            controller->m_impl->set_id(fmt::format("Channel {}", i));
            controller->m_channel_id = i;
            m_controllers.push_back(controller);
        }

        m_clock_ratio = param<uint>("clock_ratio").required();

        callback = std::bind(&AiMDRAMSystem::receive, this, std::placeholders::_1);

        register_stat(m_clk).name("memory_system_cycles");
        register_stat(s_wait_RD_stall)
            .name("total_num_wait_read_stalls")
            .desc("total number of cycles that AiM DMA is stalled because of waiting for read operations from channels");

        register_stat(s_ISR_queue_full)
            .name("total_num_ISR_full")
            .desc("total number of cycles that AiM DMA does not receive ISR because of lack of enough ISR space");

        for (const auto type : {Type::Read, Type::Write}) {
            for (const auto mem_access_region : {MemAccessRegion::MEM}) {
                s_num_RW_requests[type][mem_access_region] = 0;
                register_stat(s_num_RW_requests[type][mem_access_region])
                    .name(fmt::format("total_num_{}_{}_requests",
                                      AiMISRInfo::convert_type_to_str(type),
                                      AiMISRInfo::convert_mem_access_region_to_str(mem_access_region)))
                    .desc(fmt::format("total number of {} {} requests",
                                      AiMISRInfo::convert_type_to_str(type),
                                      AiMISRInfo::convert_mem_access_region_to_str(mem_access_region)));
            }
        }
        for (int opcode = (int)Opcode::MIN + 1; opcode < (int)Opcode::MAX; opcode++) {
            s_num_AiM_requests[(Opcode)opcode] = 0;
            register_stat(s_num_AiM_requests[(Opcode)opcode])
                .name(fmt::format("total_num_AiM_{}_requests", AiMISRInfo::convert_AiM_opcode_to_str((Opcode)opcode)))
                .desc(fmt::format("total number of AiM {} requests", AiMISRInfo::convert_AiM_opcode_to_str((Opcode)opcode)));
        }
        register_stat(s_net_busy_cycles)
            .name("net_busy_cycles")
            .desc("total NIC-lane occupancy advanced by ISR_NET_DELAY (for U_NET)");
    };

    void setup(IFrontEnd *frontend, IMemorySystem *memory_system) override {}

    bool send(Request req) override {

        if (request_queue.size() == ISR_SIZE) {
            s_ISR_queue_full++;
            return false;
        }
        request_queue.push(req);
        // m_logger->info("[CLK {}] {} pushed to the queue!", m_clk, req.str());

        switch (req.type) {
        case Type::AIM: {
            s_num_AiM_requests[req.opcode]++;
            break;
        }
        case Type::Read:
        case Type::Write: {
            s_num_RW_requests[req.type][req.mem_access_region]++;
            break;
        }
        default: {
            throw ConfigurationError("AiMDRAMSystem: unknown request type {}!", (int)req.type);
            break;
        }
        }

        return true;
    };

    void tick() override {

        bool was_AiM_request_remaining = false;
        bool is_AiM_request_remaining = false;
        for (int channel_id = 0; channel_id < MAX_CHANNEL_COUNT; channel_id++) {
            while (remaining_AiM_requests[channel_id].empty() == false) {
                was_AiM_request_remaining = true;
                // m_logger->info("[CLK {}] 0- Sending {} to channel {}", m_clk, remaining_AiM_requests[channel_id].front().str(), channel_id);
                if (m_controllers[channel_id]->send(remaining_AiM_requests[channel_id].front()) == false) {
                    // m_logger->info("[CLK {}] 0- failed", m_clk, channel_id);
                    is_AiM_request_remaining = true;
                    break;
                }
                remaining_AiM_requests[channel_id].pop();
            }
        }

        if (stalled_AiM_requests == 0) {
            if (was_AiM_request_remaining == true) {
                if (is_AiM_request_remaining == false) {
                    Request host_req = request_queue.front();
                    if (host_req.callback)
                        host_req.callback(host_req);
                    request_queue.pop();
                }
            } else if (request_queue.empty() == false) {
                // MULTI-CHANNEL ISSUE: real HBM2 has an independent command/address path PER channel,
                // so up to #channels ISRs can be issued per cycle. Issue ops to their channels until the
                // queue drains, an op backpressures (target channel buffer full -> head-of-line stop), or
                // we hit the per-cycle issue width. (Was 1 op/cycle GLOBALLY -> ~1 cmd/cycle, 98% of the
                // channels idle -- the dominant remaining modeling artifact after the all-bank fix.)
                int issue_width = (int)m_controllers.size();
                for (int issued = 0; issued < issue_width && request_queue.empty() == false; issued++) {
                Request host_req = request_queue.front();
                // m_logger->info("[CLK {}] Decoding {}...", m_clk, host_req.str());
                bool all_AiM_requests_sent = true;

                switch (host_req.type) {
                case Type::AIM: {
                    Opcode opcode = host_req.opcode;
                    auto opsize = host_req.opsize;
                    int64_t ch_mask = host_req.channel_mask;
                    // GGM_EXTEND/REDUCE use channel_mask as a DIRECT channel index (one channel per
                    // op), consistent with the plain R/W MEM path — supports up to MAX_CHANNEL_COUNT
                    // channels without the 32/64-bit bitmask limit.
                    Request aim_req = host_req;
                    switch (opcode) {

                    case Opcode::ISR_SYNC: {
                        aim_req.callback = callback;
                        for (int channel_id = 0; channel_id < m_controllers.size(); channel_id++) {
                            aim_req.AiM_req_id = AiM_req_id++;
                            aim_req.host_req_id = host_req.host_req_id;
                            // m_logger->info("[CLK {}] 2- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                            if (m_controllers[channel_id]->send(aim_req) == false) {
                                remaining_AiM_requests[channel_id].push(aim_req);
                                all_AiM_requests_sent = false;
                            }
                            stalled_AiM_requests += 1;
                        }
                        break;
                    } break;

                    case Opcode::ISR_EOC: {
                        aim_req.callback = callback;
                        for (int channel_id = 0; channel_id < m_controllers.size(); channel_id++) {
                            aim_req.AiM_req_id = AiM_req_id++;
                            aim_req.host_req_id = host_req.host_req_id;
                            // m_logger->info("[CLK {}] 3- Sending {} to channel {}", m_clk, aim_req.str(), channel_id);
                            if (m_controllers[channel_id]->send(aim_req) == false) {
                                remaining_AiM_requests[channel_id].push(aim_req);
                                all_AiM_requests_sent = false;
                            }
                            stalled_AiM_requests += 1;
                        }
                        break;
                    } break;

                    case Opcode::ISR_GGM_EXTEND: {
                        // DPF/GGM ChaCha tree-expansion (timing model):
                        //   load    : [opsize] column reads from row_in  (parent seed words)
                        //   compute : ChaCha8 (compute_latency FPU cycles, charged in the controller)
                        //   store   : [2*opsize] column writes to row_out (child seed words; 1:2 ratio)
                        // Issued as plain Type::Read/Write MEM sub-requests (reuse RD/WR commands,
                        // scheduling and row-buffer modeling) but keep opcode = ISR_GGM_EXTEND so
                        // cycle stats attribute to the GGM op (see AiMDRAMController).
                        if (opsize == -1)
                            opsize = 1;
                        int32_t row_in = host_req.row_addr;
                        int32_t row_out = host_req.row_out;
                        int32_t ggm_compute_latency = host_req.compute_latency;
                        int32_t col_in_base = (host_req.col_in >= 0) ? host_req.col_in : 0;
                        int32_t col_out_base = (host_req.col_out >= 0) ? host_req.col_out : 0;
                        // Double-buffering: store to write_bank if given, else the parent bank.
                        int16_t store_bank = (host_req.write_bank >= 0) ? host_req.write_bank
                                                                        : host_req.bank_index;
                        int channel_id = (int)ch_mask;   // direct channel index
                        if (channel_id < 0 || channel_id >= (int)m_controllers.size()) {
                            m_logger->error("ISR_GGM_EXTEND channel {} out of range (have {})!", channel_id, m_controllers.size());
                            exit(-1);
                        }
                        // ALL-BANK mode (nbanks>1): emit the opsize reads + 2*opsize writes with bank=-1
                        // so apply_addr_mapp makes them CHANNEL-scoped; the controller routes them to the
                        // ACT16/ABRD/ABWR all-bank commands (one bus event, all banks in parallel). The
                        // sub-request keeps nbanks (copied from host_req) as the controller's trigger.
                        // nbanks<=1: the original single-bank RD/WR path (baseline; bank-true addressing).
                        bool all_bank = (host_req.nbanks > 1);
                        int16_t rbank = all_bank ? (int16_t)-1 : host_req.bank_index;
                        int16_t wbank = all_bank ? (int16_t)-1 : store_bank;
                        for (int i = 0; i < opsize; i++) {
                            Request rd = host_req;
                            rd.type = Type::Read;
                            rd.mem_access_region = MemAccessRegion::MEM;
                            rd.opcode = Opcode::ISR_GGM_EXTEND;
                            rd.bank_index = rbank;
                            rd.row_addr = row_in;
                            rd.col_addr = col_in_base + i;
                            rd.compute_latency = (i == opsize - 1) ? ggm_compute_latency : -1;
                            rd.AiM_req_id = AiM_req_id++;
                            rd.host_req_id = host_req.host_req_id;
                            apply_addr_mapp(rd, channel_id);
                            rd.addr = (((int64_t)rbank << 40) | ((int64_t)row_in << 12) | rd.col_addr);
                            if (m_controllers[channel_id]->send(rd) == false) {
                                remaining_AiM_requests[channel_id].push(rd);
                                all_AiM_requests_sent = false;
                            }
                        }
                        for (int i = 0; i < 2 * opsize; i++) {
                            Request wr = host_req;
                            wr.type = Type::Write;
                            wr.mem_access_region = MemAccessRegion::MEM;
                            wr.opcode = Opcode::ISR_GGM_EXTEND;
                            wr.bank_index = wbank;
                            wr.row_addr = row_out;
                            wr.col_addr = col_out_base + i;
                            wr.compute_latency = -1;
                            wr.AiM_req_id = AiM_req_id++;
                            wr.host_req_id = host_req.host_req_id;
                            apply_addr_mapp(wr, channel_id);
                            wr.addr = (((int64_t)wbank << 40) | ((int64_t)row_out << 12) | wr.col_addr);
                            if (m_controllers[channel_id]->send(wr) == false) {
                                remaining_AiM_requests[channel_id].push(wr);
                                all_AiM_requests_sent = false;
                            }
                        }
                        break;
                    }

                    case Opcode::ISR_GGM_REDUCE: {
                        // DPF leaf reduction: read [opsize] leaf column-words (from bank_index/row_addr),
                        // apply the per-leaf 62-bit modular multiply ([compute_latency], charged on the
                        // last read), then write [wrsize] result column-words to [write_bank]/[row_out]
                        // (the scatter into g, or the 2N->N fold). Read:write ratio is configurable.
                        if (opsize == -1)
                            opsize = 1;
                        int32_t leaf_row = host_req.row_addr;
                        int32_t g_row = host_req.row_out;
                        int32_t modmul_latency = host_req.compute_latency;
                        int32_t col_in_base = (host_req.col_in >= 0) ? host_req.col_in : 0;
                        int32_t col_out_base = (host_req.col_out >= 0) ? host_req.col_out : 0;
                        int32_t g_writes = (host_req.wrsize >= 0) ? host_req.wrsize : 0;
                        int16_t g_bank = (host_req.write_bank >= 0) ? host_req.write_bank
                                                                    : host_req.bank_index;
                        int channel_id = (int)ch_mask;   // direct channel index
                        if (channel_id < 0 || channel_id >= (int)m_controllers.size()) {
                            m_logger->error("ISR_GGM_REDUCE channel {} out of range (have {})!", channel_id, m_controllers.size());
                            exit(-1);
                        }
                        // ALL-BANK mode (nbanks>1): same as GGM_EXTEND — bank=-1 -> channel-scoped ->
                        // controller routes to ABRD/ABWR (one bus event, all banks parallel).
                        bool all_bank = (host_req.nbanks > 1);
                        int16_t rbank = all_bank ? (int16_t)-1 : host_req.bank_index;
                        int16_t wbank = all_bank ? (int16_t)-1 : g_bank;
                        for (int i = 0; i < opsize; i++) {
                            Request rd = host_req;
                            rd.type = Type::Read;
                            rd.mem_access_region = MemAccessRegion::MEM;
                            rd.opcode = Opcode::ISR_GGM_REDUCE;
                            rd.bank_index = rbank;
                            rd.row_addr = leaf_row;
                            rd.col_addr = col_in_base + i;
                            rd.compute_latency = (i == opsize - 1) ? modmul_latency : -1;
                            rd.is_dru = (modmul_latency == 0);  // cl<0 = full-timing stream (no FPU, normal rd/wr buffers)
                            rd.AiM_req_id = AiM_req_id++;
                            rd.host_req_id = host_req.host_req_id;
                            apply_addr_mapp(rd, channel_id);
                            rd.addr = (((int64_t)rbank << 40) | ((int64_t)leaf_row << 12) | rd.col_addr);
                            if (m_controllers[channel_id]->send(rd) == false) {
                                remaining_AiM_requests[channel_id].push(rd);
                                all_AiM_requests_sent = false;
                            }
                        }
                        for (int i = 0; i < g_writes; i++) {
                            Request wr = host_req;
                            wr.type = Type::Write;
                            wr.mem_access_region = MemAccessRegion::MEM;
                            wr.opcode = Opcode::ISR_GGM_REDUCE;
                            wr.bank_index = wbank;
                            wr.row_addr = g_row;
                            wr.col_addr = col_out_base + i;
                            wr.compute_latency = -1;
                            wr.is_dru = (modmul_latency == 0);
                            wr.AiM_req_id = AiM_req_id++;
                            wr.host_req_id = host_req.host_req_id;
                            apply_addr_mapp(wr, channel_id);
                            wr.addr = (((int64_t)wbank << 40) | ((int64_t)g_row << 12) | wr.col_addr);
                            if (m_controllers[channel_id]->send(wr) == false) {
                                remaining_AiM_requests[channel_id].push(wr);
                                all_AiM_requests_sent = false;
                            }
                        }
                        break;
                    }

                    case Opcode::ISR_NET_DELAY: {
                        // Advance the NIC-lane timeline: opsize rungs of
                        // compute_latency CK each (one rung = one party-serialized
                        // round trip; rungs serialize on the single lane).
                        Clk_t adv = (Clk_t)host_req.opsize * (Clk_t)host_req.compute_latency;
                        m_net_ready_at = std::max((Clk_t)m_clk, m_net_ready_at) + adv;
                        s_net_busy_cycles += adv;
                        break;   // sends nothing -> popped this cycle (non-blocking)
                    }

                    case Opcode::ISR_NET_WAIT: {
                        // Network barrier: hold the queue head (break-without-pop
                        // via all_AiM_requests_sent) until the NIC lane drains.
                        if ((Clk_t)m_clk < m_net_ready_at)
                            all_AiM_requests_sent = false;
                        break;
                    }

                    default:
                        m_logger->error("unknown command \n");
                        break;
                    }
                } break;
                case Type::Read:
                case Type::Write: {
                    // Plain memory accesses (used by the GGM load/store phases). Only MEM region.
                    if (host_req.mem_access_region != MemAccessRegion::MEM) {
                        throw ConfigurationError("AiMDRAMSystem: only MEM region supported (got {})!", (int)host_req.mem_access_region);
                    }
                    Request aim_req = host_req;
                    aim_req.AiM_req_id = AiM_req_id++;
                    apply_addr_mapp(aim_req, aim_req.channel_mask);
                    int channel_id = aim_req.addr_vec[m_dram->m_levels("channel")];
                    if (m_controllers[channel_id]->send(aim_req) == false) {
                        remaining_AiM_requests[channel_id].push(aim_req);
                        all_AiM_requests_sent = false;
                    }
                    break;
                }
                default: {
                    throw ConfigurationError("AiMDRAMSystem: unknown request type {}!", (int)host_req.type);
                    break;
                }
                }
                if ((stalled_AiM_requests == 0) && (all_AiM_requests_sent == true)) {
                    if (host_req.callback)
                        host_req.callback(host_req);
                    request_queue.pop();
                } else {
                    break;   // backpressure (channel full) or SYNC/EOC stall -> stop issuing this cycle
                }
                }   // end multi-channel issue loop
            }
        }

        if (m_clk % m_controllers[0]->get_clock_ratio() == 0) {
            m_dram->tick();
            for (auto controller : m_controllers) {
                controller->tick();
            }
        }

        m_clk++;
    };

    void receive(Request &req) {
        Request host_req = request_queue.front();
        if (req.host_req_id != host_req.host_req_id)
            throw ConfigurationError("AiMDRAMSystem: received request id {} != head of the queue request id {}!", req.host_req_id, host_req.host_req_id);

        stalled_AiM_requests--;

        if (stalled_AiM_requests == 0) {
            if (host_req.callback)
                host_req.callback(host_req);
            request_queue.pop();
        }
    }

    float get_tCK() override {
        return m_dram->m_timing_vals("tCK_ps") / 1000.0f;
    }

    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };
};

} // namespace Ramulator
