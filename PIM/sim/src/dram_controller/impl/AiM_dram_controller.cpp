#include "base/request.h"
#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"
#include <cstdio>
#include <string>

namespace Ramulator {

class AiMDRAMController final : public IDRAMController, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, AiMDRAMController, "AiM", "AiM DRAM controller.");

private:
    std::deque<Request> pending_reads;   // A queue for read requests that are about to finish (callback after RL)
    std::vector<Request> pending_writes; // A queue for write requests that are about to finish

    ReqBuffer m_active_buffer;   // Buffer for requests being served. This has the highest priority
    ReqBuffer m_priority_buffer; // Buffer for high-priority requests (e.g., maintenance like refresh).
    ReqBuffer m_read_buffer;     // Read request buffer
    ReqBuffer m_write_buffer;    // Write request buffer
    ReqBuffer m_aim_buffer;      // AiM request buffer
    // Dedicated DRU stream FIFO (is_dru sub-requests: host GGM_REDUCE cl<=0).
    // The DRU consumes columns strictly in order, so this path is served
    // head-of-line in O(1) and never enters the FRFCFS-scanned buffers --
    // the stream can no longer make the per-cycle get_best_request scan
    // quadratic in the number of in-flight DRU columns.
    ReqBuffer m_dru_buffer;
    // DRU is a channel-level consumer with its own GIO/row-buffer tap: it does
    // NOT manage bank rows (the bank PU's EXTEND owns ACT/PRE) and does NOT
    // contend for the bank command scheduler. It occupies only the channel
    // DATA BUS, one column per nBL cycles, and its reads return at the column
    // read latency. This is what decouples it from EXTEND's row management
    // (no ACT/PRE thrash on the shared 2-bank ping-pong).
    Clk_t m_dru_databus_free = 0;
    int   m_dru_col_latency = 22;  // ~nCL+nBL column read latency (approx; DRU is throughput-bound)
    int   m_dru_bus_slot    = 2;   // nBL: one 32B column occupies the data bus 2 cycles
    std::deque<Request> m_dru_pending;   // DRU columns in flight (own depart queue)

    int m_row_addr_idx = -1;

    float m_wr_low_watermark;
    float m_wr_high_watermark;
    bool m_is_write_mode = false;

    // ISR_GGM_EXTEND compute model: true → ChaCha compute overlaps memory (op finishes at
    // max(mem, compute)); false → serialized (mem + compute). Set via controller param.
    bool m_ggm_compute_overlap = true;

    // Each PIM unit has ONE ChaCha FPU shared by m_banks_per_pim_unit banks (HBM-PIM: 2). Compute
    // from ops on the same PIM unit SERIALIZES on that FPU. Tracks the per-unit FPU-free time.
    int m_banks_per_pim_unit = 2;
    std::map<int, Clk_t> m_fpu_busy_until;
    // FPU initiation interval: cycles before the FPU can accept the NEXT op. -1 (default) = the full
    // compute_latency (NON-pipelined: one op occupies the FPU for its whole latency). A smaller II
    // models a PIPELINED / balanced FPU (result latency unchanged, but the next op starts after II) ->
    // consecutive ops' compute overlaps, so a compute-heavy op no longer serializes the channel.
    int m_fpu_init_interval = -1;
    // Register files per bank-group (each holds one in-flight op's state). A finite RF = a finite
    // compute-pipeline depth: at most N_RF ops can be in flight, so the FPU cannot accept a new op
    // faster than ceil(latency / N_RF). -1 (default) = unlimited (current behaviour). This makes the
    // reported eff_BW honest for a given register budget (the all-bank op occupies one RF slot per BG).
    int m_rf_per_bg = -1;

    // GGM FPU issue-gating: when true, a GGM op carrying the compute charge will NOT issue its
    // final (compute-triggering) command while that PIM-unit's FPU is still busy (m_fpu_busy_until).
    // This models a single non-pipelined ALU whose THROUGHPUT is the compute rate (one batch's 274
    // instructions must finish before the next batch's compute starts) -> compute-bound steady state.
    // Default false preserves the existing (memory-bound, compute-as-latency) behaviour.
    bool m_fpu_gate_issue = false;

    std::vector<IControllerPlugin *> m_plugins;

    size_t s_num_row_hits = 0;
    size_t s_num_row_misses = 0;
    size_t s_num_row_conflicts = 0;

    std::map<Type, int> s_num_RW_cycles;
    std::map<Opcode, int> s_num_AiM_cycles;
    std::map<int, int> s_num_commands;
    int s_num_idle_cycles = 0;
    int s_num_active_cycles = 0;
    int s_num_precharged_cycles = 0;

    bool is_reg_RW_mode = false;

    // ---- GPU-PIM contention co-simulation ----------------------------------
    // Traffic classes (Request::req_class): 0 PIM, 1 GPU, 2 DRU, 3 AGG.
    static constexpr int NCLS = 4;
    std::vector<int> m_class_prio = {0, 1, 3, 2};   // highest priority first
    int m_class_min_run = 0;
    bool m_pim_strict = false;
    size_t s_cls_wq_count[4] = {};
    double s_cls_wq_mean[4] = {};
    double m_wqsum[4] = {};     // keep serving the last-served class this many cycles
    int m_last_class = -1;
    Clk_t m_last_class_until = 0;
    int64_t m_watchdog_cycles = 5000000;
    Clk_t m_last_issue_clk = 0;
    // per-class statistics
    size_t s_cls_rd[NCLS] = {}, s_cls_wr[NCLS] = {}, s_cls_bus_cycles[NCLS] = {};
    size_t s_cls_row_hit[NCLS] = {}, s_cls_row_miss[NCLS] = {}, s_cls_row_conflict[NCLS] = {};
    size_t s_cls_q_count[NCLS] = {}, s_cls_r_count[NCLS] = {};
    double s_cls_q_mean[NCLS] = {}, s_cls_q_p50[NCLS] = {}, s_cls_q_p95[NCLS] = {}, s_cls_q_p99[NCLS] = {};
    double s_cls_r_mean[NCLS] = {}, s_cls_r_p50[NCLS] = {}, s_cls_r_p95[NCLS] = {}, s_cls_r_p99[NCLS] = {};
    size_t s_ab_col = 0, s_ab_slot_cycles = 0;
    static constexpr int HB = 4;          // histogram bucket width (cycles)
    static constexpr int NHB = 4096;      // last bucket collects everything above
    std::vector<uint64_t> m_qhist[NCLS], m_rhist[NCLS];
    double m_qsum[NCLS] = {}, m_rsum[NCLS] = {};

public:
    void init() override {
        m_wr_low_watermark = param<float>("wr_low_watermark").desc("Threshold for switching back to read mode.").default_val(0.2f);
        m_wr_high_watermark = param<float>("wr_high_watermark").desc("Threshold for switching to write mode.").default_val(0.8f);
        m_ggm_compute_overlap = param<bool>("ggm_compute_overlap").desc("ISR_GGM_EXTEND: overlap ChaCha compute with memory (max) vs serialize (sum).").default_val(true);
        m_banks_per_pim_unit = param<int>("banks_per_pim_unit").desc("Banks sharing one ChaCha FPU (HBM-PIM: 2).").default_val(2);
        m_fpu_init_interval = param<int>("fpu_init_interval").desc("FPU initiation interval (cycles to accept next op); -1 = non-pipelined (= compute_latency).").default_val(-1);
        m_rf_per_bg = param<int>("rf_per_bg").desc("Register files per bank-group (= max in-flight compute ops). -1 = unlimited.").default_val(-1);
        m_fpu_gate_issue = param<bool>("fpu_gate_issue").desc("Gate GGM op issue on FPU-busy (single-ALU compute-bound throughput).").default_val(false);
        m_clock_ratio = param<uint>("clock_ratio").required();
        {
            std::string pr = param<std::string>("class_priority")
                .desc("Comma list of traffic classes, highest first (0 PIM, 1 GPU, 2 DRU, 3 AGG).")
                .default_val("0,1,3,2");
            m_class_prio.clear();
            for (size_t i = 0; i < pr.size();) {
                size_t j = pr.find(',', i);
                if (j == std::string::npos) j = pr.size();
                m_class_prio.push_back(std::stoi(pr.substr(i, j - i)));
                i = j + 1;
            }
        }
        m_pim_strict = param<bool>("pim_strict")
            .desc("While any PIM request is queued or active, only PIM may use the channel.")
            .default_val(false);
        m_class_min_run = param<int>("class_min_run")
            .desc("Cycles to keep serving the last-served class while it has ready requests.")
            .default_val(0);
        m_watchdog_cycles = param<int64_t>("watchdog_cycles")
            .desc("Abort if requests are pending but no command issues for this many cycles.")
            .default_val(5000000);

        m_scheduler = create_child_ifce<IScheduler>();
        m_refresh = create_child_ifce<IRefreshManager>();

        if (m_config["plugins"]) {
            YAML::Node plugin_configs = m_config["plugins"];
            for (YAML::iterator it = plugin_configs.begin(); it != plugin_configs.end(); ++it) {
                m_plugins.push_back(create_child_ifce<IControllerPlugin>(*it));
            }
        }
    };

    void setup(IFrontEnd *frontend, IMemorySystem *memory_system) override {
        m_dram = memory_system->get_ifce<IDRAM>();
        m_row_addr_idx = m_dram->m_levels("row");
        m_priority_buffer.max_size = 512 * 3 + 32;
        // The default R/W buffers (32) are tiny for a PIM that bursts opsize+2*opsize column commands
        // per all-bank op: a channel fills after ~1-2 ops and backpressures the single in-order DMA
        // dispatcher, which then blocks GLOBALLY (head-of-line) -> ~1 cmd/cycle, 98% channel idle.
        // Real HBM2 channels have independent command paths + deep per-channel queues; size them so a
        // channel absorbs bursts and the dispatcher feeds all channels instead of stalling on one.
        int rwq = param<int>("req_buffer_size").desc("Per-channel read/write request-buffer depth.").default_val(2048);
        m_read_buffer.max_size = rwq;
        m_write_buffer.max_size = rwq;
        m_dru_buffer.max_size = 128;  // DRU stream FIFO
        m_logger = Logging::create_logger("AiMDRAMController[" + std::to_string(m_channel_id) + "]");

        for (const auto type : {Type::Read, Type::Write}) {
            s_num_RW_cycles[type] = 0;
            register_stat(s_num_RW_cycles[type])
                .name(fmt::format("CH{}_{}_cycles",
                                  m_channel_id,
                                  AiMISRInfo::convert_type_to_str(type)));
        }

        for (int opcode = (int)Opcode::MIN + 1; opcode < (int)Opcode::MAX; opcode++) {
            s_num_AiM_cycles[(Opcode)opcode] = 0;
            register_stat(s_num_AiM_cycles[(Opcode)opcode])
                .name(fmt::format("CH{}_AiM_{}_cycles", m_channel_id, AiMISRInfo::convert_AiM_opcode_to_str((Opcode)opcode)))
                .desc(fmt::format("total number of AiM {} cycles", AiMISRInfo::convert_AiM_opcode_to_str((Opcode)opcode)));
        }

        for (int command_id = 0; command_id < m_dram->m_commands.size(); command_id++) {
            s_num_commands[command_id] = 0;
            register_stat(s_num_commands[command_id])
                .name(fmt::format("CH{}_num_{}_commands", m_channel_id, std::string(m_dram->m_commands(command_id))))
                .desc(fmt::format("total number of {} commands", std::string(m_dram->m_commands(command_id))));
        }

        static const char* CN[NCLS] = {"pim", "gpu", "dru", "agg"};
        for (int c = 0; c < NCLS; c++) {
            m_qhist[c].assign(NHB + 1, 0);
            m_rhist[c].assign(NHB + 1, 0);
            register_stat(s_cls_rd[c]).name(fmt::format("CH{}_{}_rd_cols", m_channel_id, CN[c]));
            register_stat(s_cls_wr[c]).name(fmt::format("CH{}_{}_wr_cols", m_channel_id, CN[c]));
            register_stat(s_cls_bus_cycles[c]).name(fmt::format("CH{}_{}_bus_cycles", m_channel_id, CN[c]));
            register_stat(s_cls_row_hit[c]).name(fmt::format("CH{}_{}_row_hit", m_channel_id, CN[c]));
            register_stat(s_cls_row_miss[c]).name(fmt::format("CH{}_{}_row_miss", m_channel_id, CN[c]));
            register_stat(s_cls_row_conflict[c]).name(fmt::format("CH{}_{}_row_conflict", m_channel_id, CN[c]));
            register_stat(s_cls_q_count[c]).name(fmt::format("CH{}_{}_q_count", m_channel_id, CN[c]));
            register_stat(s_cls_q_mean[c]).name(fmt::format("CH{}_{}_q_mean", m_channel_id, CN[c]));
            register_stat(s_cls_q_p50[c]).name(fmt::format("CH{}_{}_q_p50", m_channel_id, CN[c]));
            register_stat(s_cls_q_p95[c]).name(fmt::format("CH{}_{}_q_p95", m_channel_id, CN[c]));
            register_stat(s_cls_q_p99[c]).name(fmt::format("CH{}_{}_q_p99", m_channel_id, CN[c]));
            register_stat(s_cls_wq_count[c]).name(fmt::format("CH{}_{}_wq_count", m_channel_id, CN[c]));
            register_stat(s_cls_wq_mean[c]).name(fmt::format("CH{}_{}_wq_mean", m_channel_id, CN[c]));
            register_stat(s_cls_r_count[c]).name(fmt::format("CH{}_{}_rdlat_count", m_channel_id, CN[c]));
            register_stat(s_cls_r_mean[c]).name(fmt::format("CH{}_{}_rdlat_mean", m_channel_id, CN[c]));
            register_stat(s_cls_r_p50[c]).name(fmt::format("CH{}_{}_rdlat_p50", m_channel_id, CN[c]));
            register_stat(s_cls_r_p95[c]).name(fmt::format("CH{}_{}_rdlat_p95", m_channel_id, CN[c]));
            register_stat(s_cls_r_p99[c]).name(fmt::format("CH{}_{}_rdlat_p99", m_channel_id, CN[c]));
        }
        register_stat(s_ab_col).name(fmt::format("CH{}_pim_allbank_cols", m_channel_id));
        register_stat(s_ab_slot_cycles).name(fmt::format("CH{}_pim_allbank_slot_cycles", m_channel_id));

        register_stat(s_num_idle_cycles)
            .name(fmt::format("CH{}_idle_cycles", m_channel_id))
            .desc(fmt::format("total number of idle cycles"));

        register_stat(s_num_active_cycles)
            .name(fmt::format("CH{}_active_cycles", m_channel_id))
            .desc(fmt::format("total number of active cycles"));

        register_stat(s_num_precharged_cycles)
            .name(fmt::format("CH{}_precharged_cycles", m_channel_id))
            .desc(fmt::format("total number of precharged cycles"));
    };

    // Flat PIM-unit id for a request's bank (pseudochannel*bg*bank flattened, then grouped by
    // m_banks_per_pim_unit). Used to serialize ChaCha compute on the per-unit FPU.
    int pim_unit_of(const AddrVec_t &av) {
        int bg = av[m_dram->m_levels("bankgroup")];
        int bk = av[m_dram->m_levels("bank")];
        int nbk = m_dram->get_level_size("bank");
        int pch = m_dram->m_levels.contains("pseudochannel") ? av[m_dram->m_levels("pseudochannel")] : 0;
        int nbg = m_dram->get_level_size("bankgroup");
        int flat_bank = (pch * nbg + bg) * nbk + bk;
        return flat_bank / (m_banks_per_pim_unit > 0 ? m_banks_per_pim_unit : 1);
    }

    // Single-ALU throughput gate: a GGM op's compute-triggering sub-request (the one carrying
    // compute_latency) must wait until its PIM-unit FPU is free. Returns false (= stall) only when
    // gating is enabled and the FPU for this op's unit is still busy. All non-compute sub-requests
    // (loads/ACT) and non-GGM requests are never gated.
    bool ggm_fpu_ready(const Request &r) {
        if (!m_fpu_gate_issue || r.compute_latency <= 0)
            return true;
        if ((r.opcode != Opcode::ISR_GGM_EXTEND) && (r.opcode != Opcode::ISR_GGM_REDUCE))
            return true;
        int unit = (r.nbanks > 1) ? (-1 - m_channel_id) : pim_unit_of(r.addr_vec);
        auto it = m_fpu_busy_until.find(unit);
        return (it == m_fpu_busy_until.end()) || (it->second <= m_clk);
    }

    bool compare_addr_vec(Request req1, Request req2, int min_compared_level) {
        for (int level_idx = m_dram->m_levels("channel"); level_idx <= min_compared_level; level_idx++) {
            if (req1.addr_vec[level_idx] == -1)
                return true;
            if (req2.addr_vec[level_idx] == -1)
                return true;
            if (req1.addr_vec[level_idx] != req2.addr_vec[level_idx])
                return false;
        }
        return true;
    }

    bool send(Request &req) override {
        if (req.type == Type::AIM) {
            // Barriers (SYNC/EOC) wait only for PIM-class traffic; GPU/DRU
            // requests keep flowing around them.
            if (pim_rw_pending())
                return false;
            req.final_command = m_dram->m_aim_request_translations((int)req.opcode);
        } else {
            if (req.req_class == 0 && m_aim_buffer.size() != 0)
                return false;
            req.final_command = m_dram->m_request_translations((int)req.type);
            // ALL-BANK GGM sub-requests (nbanks>1, channel-scoped, bank=-1): route to the all-bank
            // column commands so one bus event drives every bank in parallel (vs. RD/WR on one bank).
            if (req.nbanks > 1) {
                if (req.type == Type::Read)
                    req.final_command = m_dram->m_commands("ABRD");
                else if (req.type == Type::Write)
                    req.final_command = m_dram->m_commands("ABWR");
            }
        }

        // Forward existing write requests to incoming read requests
        if (req.type == Type::Read) {
            auto compare_addr = [req](const Request &wreq) {
                return wreq.addr == req.addr;
            };
            if (std::find_if(m_write_buffer.begin(), m_write_buffer.end(), compare_addr) != m_write_buffer.end()) {
                // The request will depart at the next cycle
                req.depart = m_clk + 1;
                pending_reads.push_back(req);
                return true;
            }
        }

        // Else, enqueue them to corresponding buffer based on request type id
        bool is_success = false;
        req.arrive = m_clk;
        if (req.is_dru) {
            // DRU/SM sequential stream: own FIFO, skips the write-forwarding
            // scan (its rows never alias the EXTEND ping-pong regions by
            // construction) and the FRFCFS buffers.
            return m_dru_buffer.enqueue(req);
        }
        if (req.type == Type::Read) {
            is_success = m_read_buffer.enqueue(req);
        } else if (req.type == Type::Write) {
            is_success = m_write_buffer.enqueue(req);
        } else if (req.type == Type::AIM) {
            is_success = m_aim_buffer.enqueue(req);
        } else {
            throw std::runtime_error("Invalid request type!");
        }
        if (!is_success) {
            // We could not enqueue the request
            req.arrive = -1;
            return false;
        }

        return true;
    };

    bool priority_send(Request &req) override {
        if (req.type == Type::AIM)
            req.final_command = m_dram->m_aim_request_translations((int)req.opcode);
        else
            req.final_command = m_dram->m_request_translations((int)req.type);

        bool is_success = false;
        is_success = m_priority_buffer.enqueue(req);
        return is_success;
    }

    void tick() override {
        m_clk++;
        // if ((m_clk == 1) || (m_clk % 1000 == 0))
        //     m_logger->info("[CLK {}]", m_clk);

        // 1. Serve completed reads
        serve_completed_reqs();
        serve_mau_stream();

        m_refresh->tick();

        // 2. Try to find a request to serve.
        ReqBuffer::iterator req_it;
        ReqBuffer *buffer = nullptr;
        bool request_found = schedule_request(req_it, buffer);

        // // 3. Update all plugins
        // for (auto plugin : m_plugins) {
        //     plugin->update(request_found, req_it);
        // }

        // 4. Finally, issue the commands to serve the request
        if (request_found) {
            if ((req_it->opcode == Opcode::ISR_EOC) || (req_it->opcode == Opcode::ISR_SYNC)) {
                req_it->depart = m_clk;
                pending_reads.push_back(*req_it);
                buffer->remove(req_it);
                // m_logger->info("[CLK {}] EOC/SYNC ready for callback", m_clk);
            } else {

                bool requires_reg_RW_mode = false;
                if (req_it->type == Type::AIM) {
                    // m_logger->info("Checking AiM request: {}", req_it->str());
                    if (AiMISRInfo::opcode_requires_reg_RW_mod(req_it->opcode)) {
                        requires_reg_RW_mode = true;
                    }
                }

                if (requires_reg_RW_mode ^ is_reg_RW_mode) {
                    req_it->command = m_dram->m_commands("TMOD");
                    is_reg_RW_mode = !is_reg_RW_mode;
                }

                // If we find a real request to serve
                // m_logger->info("[CLK {}] Issuing {} for {}", m_clk, std::string(m_dram->m_commands(req_it->command)).c_str(), req_it->str());
                const bool first_cmd = (req_it->issue == -1);
                if (req_it->issue == -1)
                    req_it->issue = m_clk - 1;
                m_dram->issue_command(req_it->command, req_it->addr_vec);
                s_num_commands[req_it->command] += 1;
                if (buffer != &m_priority_buffer)       // refresh is not progress
                    m_last_issue_clk = m_clk;
                record_issue(*req_it, first_cmd);
                // shared channel data bus: column commands (RD/WR/ABRD/ABWR)
                // occupy it m_dru_bus_slot cycles, delaying the DRU tap.
                if (m_dram->m_command_meta(req_it->command).is_accessing)
                    m_dru_databus_free = std::max(m_dru_databus_free, (Clk_t)m_clk + m_dru_bus_slot);

                // If we are issuing the last command, set depart clock cycle and move the request to the pending_reads queue
                if (req_it->command == req_it->final_command) {
                    int latency = m_dram->m_command_latencies(req_it->command);
                    assert(latency > 0);
                    Clk_t mem_depart = m_clk + latency;
                    // ISR_GGM_EXTEND: charge the ChaCha8 compute (FPU cycles) carried on the marked
                    // sub-request. The compute runs on the per-unit FPU, which SERIALIZES ops sharing
                    // that unit (fpu_busy_until). Overlap → op completes at max(memory, compute_done);
                    // serial → mem + compute.
                    if (req_it->compute_latency > 0) {
                        // All-bank ops (nbanks>1) are channel-scoped (bank=-1), so pim_unit_of would
                        // read a bogus bank index. All banks' FPUs run in parallel = ONE ChaCha per op,
                        // so charge it on a single per-CHANNEL FPU key (negative, won't collide with the
                        // real per-unit keys >=0); consecutive all-bank ops on this channel serialize on it.
                        int unit = (req_it->nbanks > 1) ? (-1 - m_channel_id)
                                                        : pim_unit_of(req_it->addr_vec);
                        Clk_t fpu_start = std::max((Clk_t)m_clk, m_fpu_busy_until[unit]);
                        Clk_t compute_done = fpu_start + req_it->compute_latency;   // result latency
                        // Pipelined FPU: free it after the initiation interval (II), not the full
                        // latency, so the next op can start while this one is still in the pipe.
                        int ii_cfg = (m_fpu_init_interval > 0) ? m_fpu_init_interval
                                                               : req_it->compute_latency;
                        // FINITE REGISTER FILE: with N_RF register files, at most N_RF ops fit in the
                        // pipeline, so the FPU can't accept a new op faster than ceil(latency / N_RF).
                        int ii_rf = (m_rf_per_bg > 0)
                                        ? (req_it->compute_latency + m_rf_per_bg - 1) / m_rf_per_bg
                                        : ii_cfg;
                        int ii = std::min((int)req_it->compute_latency, std::max(ii_cfg, ii_rf));
                        if (m_ggm_compute_overlap) {
                            m_fpu_busy_until[unit] = fpu_start + ii;
                        } else {
                            // no-LSU model: without load/store staging the ALU
                            // idles through the op's column I/O and cannot
                            // pipeline into the next batch -- the FPU is held
                            // for the SERIAL mem+compute window, no II.
                            Clk_t serial_done =
                                std::max(fpu_start, mem_depart) +
                                req_it->compute_latency;
                            m_fpu_busy_until[unit] = serial_done;
                            compute_done = serial_done;
                        }
                        if (getenv("DPFDBG") && m_channel_id == 0) {
                            static int dbg_n = 0;
                            if (dbg_n++ < 1000000)
                                fprintf(stderr, "[DPFDBG] clk=%ld unit=%d cl=%d fpu_start=%ld busy->%ld cmd=%d final=%d nbanks=%d\n",
                                        (long)m_clk, unit, (int)req_it->compute_latency,
                                        (long)fpu_start, (long)m_fpu_busy_until[unit],
                                        (int)req_it->command, (int)req_it->final_command, (int)req_it->nbanks);
                        }
                        req_it->depart = m_ggm_compute_overlap ? std::max(mem_depart, compute_done)
                                                               : (mem_depart + req_it->compute_latency);
                    } else {
                        req_it->depart = mem_depart;
                    }
                    if (req_it->is_reader()) {
                        pending_reads.push_back(*req_it);
                    } else {
                        pending_writes.push_back(*req_it);
                    }
                    // Attribute cycles: GGM sub-requests are Type::Read/Write but keep their GGM
                    // opcode (EXTEND/REDUCE) so they count toward that op, not raw R/W.
                    if (req_it->opcode == Opcode::ISR_GGM_EXTEND || req_it->opcode == Opcode::ISR_GGM_REDUCE) {
                        s_num_AiM_cycles[req_it->opcode] += (m_clk - req_it->issue);
                    } else if (req_it->type == Type::AIM) {
                        s_num_AiM_cycles[req_it->opcode] += (m_clk - req_it->issue);
                    } else {
                        s_num_RW_cycles[req_it->type] += (m_clk - req_it->issue);
                    }
                    // else if (req_it->type == Type::Write) {
                    //     // TODO: Add code to update statistics
                    // }
                    buffer->remove(req_it);
                } else if (req_it->type != Type::AIM) {
                    if (m_dram->m_command_meta(req_it->command).is_opening) {
                        m_active_buffer.enqueue(*req_it);
                        buffer->remove(req_it);
                    }
                }
            }
        } else if (m_read_buffer.size() == 0 && m_write_buffer.size() == 0 && m_aim_buffer.size() == 0 && m_dru_buffer.size() == 0 && pending_reads.size() == 0 && pending_writes.size() == 0) {
            // if (m_channel_id == 0)
            // m_logger->info("[CLK {}] CH0 IDLE", m_clk);
            s_num_idle_cycles += 1;
        }

        if (!request_found && m_watchdog_cycles > 0 &&
            (m_read_buffer.size() || m_write_buffer.size() || m_active_buffer.size() || m_priority_buffer.size()) &&
            (int64_t)(m_clk - m_last_issue_clk) > m_watchdog_cycles) {
            throw std::runtime_error(fmt::format(
                "AiM controller CH{}: watchdog -- no command issued for {} cycles with requests pending "
                "(rd {} wr {} act {} prio {} aim {})", m_channel_id, m_clk - m_last_issue_clk,
                m_read_buffer.size(), m_write_buffer.size(), m_active_buffer.size(),
                m_priority_buffer.size(), m_aim_buffer.size()));
        }
        if (m_dram->m_open_rows[m_channel_id] == 0) {
            s_num_precharged_cycles += 1;
        } else {
            s_num_active_cycles += 1;
        }
    };

private:
    /**
     * @brief    Helper function to serve the completed read requests
     * @details
     * This function is called at the beginning of the tick() function.
     * It checks the pending_reads queue to see if the top request has received data from DRAM.
     * If so, it finishes this request by calling its callback and poping it from the pending_reads queue.
     */
    // DRU/SM stream: a channel-level GIO tap. One column per m_dru_bus_slot
    // cycles when the data bus is free (contends with EXTEND column commands
    // via m_dru_databus_free, bumped at issue). No ACT/PRE: the DRU never
    // manages bank rows, so it cannot thrash EXTEND's ping-pong rows -- which
    // is exactly the hardware two-tier split (bank PU owns rows, base-die DRU
    // taps the row buffer). Reads return at m_dru_col_latency.
    void serve_mau_stream() {
        // completion
        while (!m_dru_pending.empty() && m_dru_pending.front().depart <= m_clk) {
            Request r = m_dru_pending.front();
            m_dru_pending.pop_front();
            if (r.callback) r.callback(r);
        }
        // issue one column if the data bus is free
        if (m_dru_buffer.size() != 0 && m_clk >= m_dru_databus_free) {
            Request r = *m_dru_buffer.begin();
            m_dru_buffer.remove(m_dru_buffer.begin());
            r.depart = m_clk + m_dru_col_latency;
            s_num_commands[m_dram->m_commands(r.is_reader() ? "RD" : "WR")] += 1;
            m_dru_pending.push_back(r);
            m_dru_databus_free = m_clk + m_dru_bus_slot;
        }
    }

    void serve_completed_reqs() {
        // PIM-class (0) completions keep the original semantics: only the oldest
        // PIM entry is considered, at most one per cycle, and a barrier waits for
        // the PIM writes. Other classes complete out of order as soon as their
        // data returns -- otherwise a PIM read carrying ChaCha compute latency
        // would hold every later GPU read hostage in this FIFO.
        bool pim_considered = false;
        for (auto it = pending_reads.begin(); it != pending_reads.end();) {
            Request &req = *it;
            if (req.req_class != 0) {
                if (req.depart <= m_clk) {
                    record_read_latency(req);
                    if (req.callback) req.callback(req);
                    it = pending_reads.erase(it);
                    continue;
                }
                ++it;
                continue;
            }
            if (pim_considered) { ++it; continue; }
            pim_considered = true;
            if (req.depart <= m_clk &&
                (((req.opcode != Opcode::ISR_EOC) && (req.opcode != Opcode::ISR_SYNC)) ||
                 (pim_pending_writes() == 0))) {
                if (req.callback) req.callback(req);
                it = pending_reads.erase(it);
                continue;
            }
            ++it;
        }
        auto write_req_it = pending_writes.begin();
        while (write_req_it != pending_writes.end()) {
            if (write_req_it->depart <= m_clk) {
                if (write_req_it->req_class != 0 && write_req_it->callback)
                    write_req_it->callback(*write_req_it);   // host streams track write completion
                // Remove this write request
                // m_logger->info("[CLK {}] Finished {}!", m_clk, write_req_it->str());
                write_req_it = pending_writes.erase(write_req_it);
            } else {
                ++write_req_it;
            }
        }
    };

    /**
     * @brief    Checks if we need to switch to write mode
     * 
     */
    void set_write_mode() {
        if (!m_is_write_mode) {
            if ((m_write_buffer.size() > m_wr_high_watermark * m_write_buffer.max_size) || m_read_buffer.size() == 0) {
                m_is_write_mode = true;
            }
        } else {
            if ((m_write_buffer.size() < m_wr_low_watermark * m_write_buffer.max_size) && m_read_buffer.size() != 0) {
                m_is_write_mode = false;
            }
        }
    };

    bool pim_rw_pending() {
        for (auto &r : m_read_buffer) if (r.req_class == 0) return true;
        for (auto &r : m_write_buffer) if (r.req_class == 0) return true;
        return false;
    }
    bool pim_present() {
        if (pim_rw_pending()) return true;
        for (auto &r : m_active_buffer) if (r.req_class == 0) return true;
        return false;
    }
    size_t pim_pending_writes() const {
        size_t n = 0;
        for (const auto &r : pending_writes) if (r.req_class == 0) n++;
        return n;
    }

    // FRFCFS within a class, strict priority across classes. For class 0 this is
    // exactly the original FRFCFS pick (ready first, then oldest) followed by the
    // FPU gate, so a PIM-only run schedules identically to the unmodified model.
    // A gated or unready class falls through to the next class, which is how GPU
    // traffic uses the cycles the SPUs spend computing.
    ReqBuffer::iterator pick_by_class(ReqBuffer &buffer, bool &found) {
        found = false;
        if (buffer.size() == 0) return buffer.end();
        for (auto &req : buffer)
            req.command = m_dram->get_preq_command(req.final_command, req.addr_vec);
        auto best_of = [&](int cls) {
            auto cand = buffer.end();
            bool cand_ready = false;
            for (auto it = buffer.begin(); it != buffer.end(); ++it) {
                if (it->req_class != cls) continue;
                const bool rdy = m_dram->check_ready(it->command, it->addr_vec);
                if (cand == buffer.end() || (rdy && !cand_ready) ||
                    (rdy == cand_ready && it->arrive < cand->arrive)) {
                    cand = it;
                    cand_ready = rdy;
                }
            }
            return cand;
        };
        if (m_pim_strict && pim_present()) {
            auto it = best_of(0);
            if (it != buffer.end())
                found = m_dram->check_ready(it->command, it->addr_vec) && ggm_fpu_ready(*it);
            return it;
        }
        std::vector<int> order;
        if (m_class_min_run > 0 && m_last_class >= 0 && m_clk < m_last_class_until)
            order.push_back(m_last_class);
        for (int c : m_class_prio) if (order.empty() || c != order[0]) order.push_back(c);
        auto fallback = buffer.end();
        for (int c : order) {
            auto it = best_of(c);
            if (it == buffer.end()) continue;
            if (fallback == buffer.end()) fallback = it;
            if (m_dram->check_ready(it->command, it->addr_vec) && ggm_fpu_ready(*it)) {
                found = true;
                return it;
            }
        }
        return fallback;
    }

    void record_issue(const Request &r, bool first_cmd) {
        if (r.opcode == Opcode::ISR_EOC || r.opcode == Opcode::ISR_SYNC) return;
        const int c = (r.req_class >= 0 && r.req_class < NCLS) ? r.req_class : 0;
        const auto &meta = m_dram->m_command_meta(r.command);
        if (first_cmd && r.arrive >= 0) {
            const Clk_t q = m_clk - r.arrive;
            if (r.type == Type::Write) {       // posted writes: queued, but never stall the issuer
                m_wqsum[c] += q;
                s_cls_wq_count[c] += 1;
            } else {
                m_qhist[c][std::min<Clk_t>(q / HB, NHB)] += 1;
                m_qsum[c] += q;
                s_cls_q_count[c] += 1;
            }
            if (r.command == r.final_command) s_cls_row_hit[c] += 1;
            else if (meta.is_closing) s_cls_row_conflict[c] += 1;
            else s_cls_row_miss[c] += 1;
        }
        if (meta.is_accessing) {
            static const int ABRD = m_dram->m_commands("ABRD");
            static const int ABWR = m_dram->m_commands("ABWR");
            if (r.command == ABRD || r.command == ABWR) {
                s_ab_col += 1;
                s_ab_slot_cycles += m_dram->m_timing_vals("nCCDL");
            } else {
                if (r.type == Type::Read) s_cls_rd[c] += 1; else s_cls_wr[c] += 1;
                s_cls_bus_cycles[c] += m_dram->m_timing_vals("nBL");
            }
            if (m_class_min_run > 0 && c != 0) {
                m_last_class = c;
                m_last_class_until = m_clk + m_class_min_run;
            }
        }
    }

    void record_read_latency(const Request &r) {
        const int c = (r.req_class >= 0 && r.req_class < NCLS) ? r.req_class : 0;
        if (r.arrive < 0) return;
        const Clk_t l = m_clk - r.arrive;
        m_rhist[c][std::min<Clk_t>(l / HB, NHB)] += 1;
        m_rsum[c] += l;
        s_cls_r_count[c] += 1;
    }

    static double pct(const std::vector<uint64_t> &h, uint64_t n, double q) {
        if (n == 0) return 0.0;
        const uint64_t target = static_cast<uint64_t>(q * n);
        uint64_t acc = 0;
        for (size_t i = 0; i < h.size(); i++) {
            acc += h[i];
            if (acc > target) return (i + 0.5) * HB;
        }
        return h.size() * HB;
    }

public:
    void dump_pending(const char *tag, ReqBuffer &b) {
        int n = 0;
        for (auto &r : b) {
            if (n++ >= 4) break;
            std::string av;
            for (auto x : r.addr_vec) av += std::to_string(x) + " ";
            std::fprintf(stderr, "[CH%d %s] class %d type %d cmd %s final %s arrive %ld issue %ld addr [%s]\n",
                         m_channel_id, tag, (int)r.req_class, (int)r.type,
                         r.command >= 0 ? std::string(m_dram->m_commands(r.command)).c_str() : "-",
                         r.final_command >= 0 ? std::string(m_dram->m_commands(r.final_command)).c_str() : "-",
                         (long)r.arrive, (long)r.issue, av.c_str());
        }
    }

    void finalize() override {
        if (getenv("CTRL_DUMP") && (m_read_buffer.size() || m_write_buffer.size() || m_active_buffer.size())) {
            dump_pending("act", m_active_buffer);
            dump_pending("rd", m_read_buffer);
            dump_pending("wr", m_write_buffer);
        }
        for (int c = 0; c < NCLS; c++) {
            const uint64_t nq = s_cls_q_count[c], nr = s_cls_r_count[c];
            s_cls_q_mean[c] = nq ? m_qsum[c] / nq : 0.0;
            s_cls_q_p50[c] = pct(m_qhist[c], nq, 0.50);
            s_cls_q_p95[c] = pct(m_qhist[c], nq, 0.95);
            s_cls_q_p99[c] = pct(m_qhist[c], nq, 0.99);
            s_cls_r_mean[c] = nr ? m_rsum[c] / nr : 0.0;
            s_cls_wq_mean[c] = s_cls_wq_count[c] ? m_wqsum[c] / s_cls_wq_count[c] : 0.0;
            s_cls_r_p50[c] = pct(m_rhist[c], nr, 0.50);
            s_cls_r_p95[c] = pct(m_rhist[c], nr, 0.95);
            s_cls_r_p99[c] = pct(m_rhist[c], nr, 0.99);
        }
    }

private:
    /**
     * @brief    Helper function to find a request to schedule from the buffers.
     * 
     */
    bool schedule_request(ReqBuffer::iterator &req_it, ReqBuffer *&req_buffer) {
        bool request_found = false;
        // 2.1    First, check the act buffer to serve requests that are already activating (avoid useless ACTs)
        if (req_it = m_scheduler->get_best_request(m_active_buffer); req_it != m_active_buffer.end()) {
            if (m_dram->check_ready(req_it->command, req_it->addr_vec) && ggm_fpu_ready(*req_it)) {
                request_found = true;
                req_buffer = &m_active_buffer;
                // m_logger->info("[CLK {}] Found request in active buffer: {}", m_clk, req_it->str());
            }
        }

        // 2.2    If no requests can be scheduled from the act buffer, check the rest of the buffers
        if (!request_found) {
            // 2.2.1    We first check the priority buffer to prioritize e.g., maintenance requests
            if (m_priority_buffer.size() != 0) {
                req_buffer = &m_priority_buffer;
                req_it = m_priority_buffer.begin();
                req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);

                request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
                if ((request_found == false) & (m_priority_buffer.size() != 0)) {
                    return false;
                }
                // if (request_found) {
                //     m_logger->info("[CLK {}] Found request in priority buffer: {}", m_clk, req_it->str());
                // }
            }

            // 2.2.1    If no request to be scheduled in the priority buffer, check the read and write OR AiM buffers.
            if (!request_found) {
                if (m_aim_buffer.size() != 0) {
                    req_it = m_aim_buffer.begin();
                    if ((req_it->opcode == Opcode::ISR_EOC) || (req_it->opcode == Opcode::ISR_SYNC)) {
                        // Barrier drains the DRU FIFO within the round. This is
                        // a CONSERVATIVE no-overlap model (DRU cost fully
                        // exposed each round, an upper bound); the physical
                        // DRU-overlaps-next-round benefit is argued separately.
                        // Keeps the sim fast (bounded FIFO, no O(n) pileup).
                        if (m_dru_buffer.size() != 0) { request_found = false; }
                        else { req_buffer = &m_aim_buffer; return true; }
                    } else {
                        req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
                        request_found = m_dram->check_ready(req_it->command, req_it->addr_vec) && ggm_fpu_ready(*req_it);
                        req_buffer = &m_aim_buffer;
                        // if (request_found) {
                        //     m_logger->info("[CLK {}] 2- Found AiM request in buffer: {}", m_clk, req_it->str());
                        // }
                    }
                } else {
                    // Query the write policy to decide which buffer to serve
                    set_write_mode();
                    if (m_pim_strict && pim_present()) {
                        // PIM owns the channel: serve whichever buffer holds its requests,
                        // otherwise its writes starve behind host reads (and vice versa).
                        bool rd0 = false, wr0 = false;
                        for (auto &r : m_read_buffer) if (r.req_class == 0) { rd0 = true; break; }
                        for (auto &r : m_write_buffer) if (r.req_class == 0) { wr0 = true; break; }
                        if (m_is_write_mode && !wr0 && rd0) m_is_write_mode = false;
                        else if (!m_is_write_mode && !rd0 && wr0) m_is_write_mode = true;
                    }
                    auto &buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
                    req_it = pick_by_class(buffer, request_found);
                    if (req_it != buffer.end())
                        req_buffer = &buffer;
                }
                // DRU stream FIFO: strictly in-order head service, O(1). Fills
                // the cycles the gated EXTEND stream leaves idle (which is
                // exactly the hardware behavior: the DRU drains columns while
                // the bank PUs compute). Never reordered, never scanned.

            }
        }

        // 2.3 If we find a request to schedule, we need to check if it will close an opened row in the active buffer.
        if (request_found) {
            if (m_dram->m_command_meta(req_it->command).is_closing) {
                // Do not close a row that a request in the active buffer has opened and
                // not yet used. -1 (all-bank / channel scope) overlaps every bank, so a
                // PREA cannot interrupt a single-bank request and a PRE cannot interrupt
                // an all-bank one.
                for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end(); _it++) {
                    bool overlap = true;
                    for (int l = 0; l < m_row_addr_idx; l++) {
                        const Addr_t a = req_it->addr_vec[l], b = _it->addr_vec[l];
                        if (a != -1 && b != -1 && a != b) { overlap = false; break; }
                    }
                    if (overlap) {
                        request_found = false;
                        break;
                    }
                }
            }
        }

        return request_found;
    }
};

} // namespace Ramulator