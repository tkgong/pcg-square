#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

class HBM2 : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, HBM2, "HBM2", "HBM2 Device Model")

  public:
    inline static const std::map<std::string, Organization> org_presets = {
      //   name     density   DQ    Ch Pch  Bg Ba   Ro     Co
      {"HBM2_2Gb",   {2<<10,  128,  {1, 2,  4,  2, 1<<14, 1<<6}}},
      {"HBM2_4Gb",   {4<<10,  128,  {1, 2,  4,  4, 1<<14, 1<<6}}},
      {"HBM2_8Gb",   {6<<10,  128,  {1, 2,  4,  4, 1<<15, 1<<6}}},
      // PIM config: 64 channels, 1 rank/channel (pseudochannel=1), 1 bankgroup, 2 banks/rank.
      // With banks_per_pim_unit=2 -> ONE PU per channel = 64 PUs. Per-channel density=2048 Mb
      // (=pch*bg*ba*ro*co*dq/2^20 = 1*1*2*(1<<17)*(1<<6)*128/2^20) so density_id=0 (2Gb) is valid.
      //   name        density  DQ    Ch  Pch Bg Ba   Ro      Co
      {"PIM64_2Banks", {2<<10, 128,  {64, 1,  1, 2, 1<<17, 1<<6}}},
      // L40S GDDR6 org: 24 x16-channels (384-bit card), 4 BG x 4 banks = 16 banks/channel.
      // With banks_per_pim_unit=2 -> 8 PUs/channel = 192 PUs full card. Per-channel density
      // = 1*4*4*(1<<17)*(1<<6)*128 >> 20 = 16384 Mb = 2 GB (x24 = 48 GB = L40S), density_id=3.
      // Column geometry kept identical to PIM64 (64 cols/row, 256b PU column) so the validated
      // trace-generator address math carries over unchanged.
      //   name        density   DQ    Ch  Pch Bg Ba   Ro      Co
      {"GDDR6_L40S",  {16<<10,  128,  {24, 1,  4, 4, 1<<17, 1<<6}}},
      // A100-80GB HBM2e: sim "channel" = one physical PSEUDO-CHANNEL (16B/CK
      // = 25.6 GB/s at tCK=625ps). 5 stacks x 8ch x 2PC = 80 PCs x 16 banks
      // -> 640 PUs at 2 banks/PU; aggregate 80 x 25.6 = 2.05 TB/s = A100 SXM.
      // Per-PC density 4x4x(1<<16)x(1<<6)x128 >> 20 = 8192 Mb = 1 GB (x80 = 80GB).
      {"HBM2E_A100",  {8<<10,   128,  {80, 1,  4, 4, 1<<16, 1<<6}}},
      // B200 ONE DIE-DOMAIN (dual-die card: run 2 independent domains, x2 the
      // throughput; MAX_CHANNEL_COUNT=128 caps one sim at a domain, which is
      // also the physical partition -- no cross-domain bank traffic exists).
      // 4 stacks x 16ch x 2PC = 128 PCs x 16 banks -> 1024 PUs/domain
      // (2048/card). Per-PC 32 GB/s at tCK=500ps; 128 x 32 = 4.1 TB/s/domain.
      // Density field 8192 Mb/PC (1GB; capacity is not the object of the
      // cycle sim -- the timing/bank geometry is).
      {"HBM3E_B200D", {8<<10,   128,  {128, 1, 4, 4, 1<<16, 1<<6}}},
      // B200 FULL CARD: both die domains in one simulation. 256 pseudo-channels
      // x 16 banks, 2 banks/PU -> 2048 SPUs. Same per-channel geometry as
      // HBM3E_B200D, so the validated address math and timing carry over; only
      // the channel count doubles (requires MAX_CHANNEL_COUNT >= 256).
      {"HBM3E_B200F", {8<<10,   128,  {256, 1, 4, 4, 1<<16, 1<<6}}},
      // Hypothetical larger PIM budgets on the same HBM3e geometry, used only
      // to locate the design-space knee above the B200 channel count.
      {"HBM3E_X512",  {8<<10,   128,  {512, 1, 4, 4, 1<<16, 1<<6}}},
      {"HBM3E_X1024", {8<<10,   128, {1024, 1, 4, 4, 1<<16, 1<<6}}},
      // RTX 5000 Ada: 32GB GDDR6 256-bit = 16 x16-channels x 16 banks
      // -> 128 PUs at 2 banks/PU. Same per-channel geometry as GDDR6_L40S.
      {"GDDR6_ADA5K", {16<<10,  128,  {16, 1,  4, 4, 1<<17, 1<<6}}},
      // RTX PRO 6000 Blackwell: 96GB GDDR7 512-bit. GDDR7 die = 4 x8b channels;
      // model channel = 16B/CK unit -> 64 channels x 16 banks -> 512 PUs.
      // Density field 2GB/ch (capacity overshoot vs 1.5GB actual; cycle-neutral,
      // HBM3E_B200D precedent).
      {"GDDR7_RTX6KBW", {16<<10, 128, {64, 1,  4, 4, 1<<17, 1<<6}}},
      // H200: 6 HBM3e stacks x 16ch x 2PC = 192 PCs > MAX_CHANNEL_COUNT=128,
      // so model HALF the card (3 stacks = 96 PCs -> 768 PUs) and double the
      // throughput (channel scaling measured linear, 65.6x/64ch). 1536 PUs/card.
      {"HBM3E_H200H", {8<<10,   128,  {96, 1,  4, 4, 1<<16, 1<<6}}},
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      //   name       rate   nBL  nCL  nRCDRD  nRCDWR  nRP  nRAS  nRC  nWR  nRTPS  nRTPL  nCWL  nCCDS  nCCDL  nRRDS  nRRDL  nWTRS  nWTRL  nRTW  nFAW  nRFC  nRFCSB  nREFI  nREFISB  nRREFD  tCK_ps
      {"HBM2_2Gbps",  {2000,   4,   7,    7,      7,     7,   17,  19,   8,    2,     3,    2,    1,      2,     2,     3,     3,     4,    3,    15,   -1,   160,   3900,     -1,      8,   1000}},
      // HBM3_6400Mbps timing from upstream CMU-SAFARI ramulator2 python/ramulator/dram/hbm3.py
      // (CK units, tCK=625ps). "rate"=3200 so the bundled tCK=1e6/(rate/2) formula yields the
      // correct HBM3 command-clock tCK=625ps (data rate is 6400 MT/s = 4 beats/CK). Run on the
      // HBM2 device because GGM/AiM support lives only in HBM2.cpp; the device-wrapper difference
      // (refresh/hierarchy naming) does not affect the GGM compute-bound cycle analysis.
      //                 rate  nBL nCL nRCDRD nRCDWR nRP nRAS nRC nWR nRTPS nRTPL nCWL nCCDS nCCDL nRRDS nRRDL nWTRS nWTRL nRTW nFAW nRFC nRFCSB nREFI nREFISB nRREFD tCK_ps
      {"HBM3_6400Mbps", {3200,  2,  20,   31,    15,  26,  45,  72,  33,   9,    9,   10,    2,    4,    4,    5,    7,    10,   20,  24,  -1,  320,  6240,    -1,    8,    625}},
      // GDDR6 18 Gbps (L40S): command clock CK = 18/8 = 2.25 GHz ("rate"=4500 keeps the
      // tCK=1e6/(rate/2) convention -> tCK=444ps). nBL=2 -> one 256b PU column occupies the
      // x16 data bus 2 CK = 16B/CK = 36 GB/s/channel (x24 = 864 GB/s card peak = L40S).
      // ns->CK at 444ps: tRCD~14ns=32, tRP~14ns=32, tRAS~28ns=63, tRC~42ns=95, tWR~15ns=34,
      // tCCDL=3 (same-BG b2b), tCCDS=2 (data-bus limit), tFAW~14ns=32, tREFI~1.9us=4280.
      // Refresh fields are second-order for our us-scale measurement windows.
      //               rate  nBL nCL nRCDRD nRCDWR nRP nRAS nRC nWR nRTPS nRTPL nCWL nCCDS nCCDL nRRDS nRRDL nWTRS nWTRL nRTW nFAW nRFC nRFCSB nREFI nREFISB nRREFD tCK_ps
      {"GDDR6_18Gbps", {4500,  2,  32,   32,    20,  32,  63,  95,  34,   4,    6,    8,    2,    3,    4,    6,    6,    8,   18,  32,  -1,  248,  4280,    -1,    8,    444}},
      // HBM2e 3.2 GT/s (A100): command clock 1.6 GHz (tCK=625ps). ns->CK at
      // 625ps: tRCD~14ns=22, tRP=22, tRAS~33ns=53, tRC~47ns=75, tWR~16ns=26,
      // tFAW~16ns=26. Per-PC data path 16B/CK = 25.6 GB/s (nBL=2).
      //               rate  nBL nCL nRCDRD nRCDWR nRP nRAS nRC nWR nRTPS nRTPL nCWL nCCDS nCCDL nRRDS nRRDL nWTRS nWTRL nRTW nFAW nRFC nRFCSB nREFI nREFISB nRREFD tCK_ps
      {"HBM2E_3200",  {3200,  2,  22,   22,    11,  22,  53,  75,  26,   7,    7,    8,    2,    4,    4,    5,    6,    9,   18,  26,  -1,  256,  6240,    -1,    8,    625}},
      // HBM3e 8 GT/s (B200): command clock 2.0 GHz (tCK=500ps). ns->CK at
      // 500ps: tRCD~14ns=28, tRP=28, tRAS~28ns=56, tRC~42ns=84, tWR~15ns=30,
      // tFAW~14ns=28. Per-PC data path 16B/CK = 32 GB/s (nBL=2).
      {"HBM3E_8000",  {4000,  2,  28,   28,    14,  28,  56,  84,  30,   8,    8,    9,    2,    4,    4,    5,    6,    9,   20,  28,  -1,  280,  7800,    -1,    8,    500}},
      // GDDR7 28 Gbps (RTX PRO 6000 Blackwell): command clock 1.75 GHz
      // (tCK=571ps; rate=3500 under the tCK=1e6/(rate/2) convention). nBL=2 ->
      // 16B/CK = 28 GB/s/channel x64 = 1792 GB/s card peak. ns->CK at 571ps:
      // tRCD~14ns=25, tRP=25, tRAS~28ns=49, tRC~42ns=74, tWR~15ns=27,
      // tCCDL=3, tCCDS=2, tFAW~14ns=25.
      //               rate  nBL nCL nRCDRD nRCDWR nRP nRAS nRC nWR nRTPS nRTPL nCWL nCCDS nCCDL nRRDS nRRDL nWTRS nWTRL nRTW nFAW nRFC nRFCSB nREFI nREFISB nRREFD tCK_ps
      {"GDDR7_28Gbps", {3500,  2,  25,   25,    16,  25,  49,  74,  27,   4,    5,    7,    2,    3,    4,    5,    5,    7,   16,  25,  -1,  193,  3330,    -1,    8,    571}},
      // TODO: Find more sources on HBM2 timings...
    };


  /************************************************
   *                Organization
   ***********************************************/   
    const int m_internal_prefetch_size = 2;

    inline static constexpr ImplDef m_levels = {
      "channel", "pseudochannel", "bankgroup", "bank", "row", "column",    
    };


  /************************************************
   *             Requests & Commands
   ***********************************************/
    inline static constexpr ImplDef m_commands = {
      "ACT",
      "PRE", "PREA",
      "RD",  "WR",  "RDA",  "WRA",
      // All-bank PIM commands (one bus event, executed by ALL banks of the channel in parallel —
      // the SK-Hynix-AiM / Samsung-HBM-PIM all-bank mode). ACT16 = all-bank activate, ABRD/ABWR =
      // all-bank column read/write. (ACT16 name reused from GDDR6 so the existing all-bank lambdas
      // Channel::ACTab / RequireAllRowsOpen apply unchanged.)
      "ACT16", "ABRD", "ABWR",
      "REFab", "REFsb",
      // AiM control commands (no data movement; used by the AiM frontend/controller).
      "TMOD", "SYNC", "EOC", "UNKNOWN"
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT",   "row"},
        {"PRE",   "bank"},    {"PREA",   "channel"},
        {"RD",    "column"},  {"WR",     "column"}, {"RDA",   "column"}, {"WRA",   "column"},
        {"ACT16", "channel"}, {"ABRD",   "channel"}, {"ABWR",  "channel"},
        {"REFab", "channel"}, {"REFsb",  "bank"},
        {"TMOD",  "channel"}, {"SYNC",   "channel"}, {"EOC", "channel"}, {"UNKNOWN", "channel"},
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
                // open?   close?   access?  refresh?
        {"ACT",   {true,   false,   false,   false}},
        {"PRE",   {false,  true,    false,   false}},
        {"PREA",  {false,  true,    false,   false}},
        {"RD",    {false,  false,   true,    false}},
        {"WR",    {false,  false,   true,    false}},
        {"RDA",   {false,  true,    true,    false}},
        {"WRA",   {false,  true,    true,    false}},
        {"ACT16", {true,   false,   false,   false}},   // all-bank activate (is_opening)
        {"ABRD",  {false,  false,   true,    false}},   // all-bank read   (is_accessing)
        {"ABWR",  {false,  false,   true,    false}},   // all-bank write  (is_accessing)
        {"REFab", {false,  false,   false,   true }},
        {"REFsb", {false,  false,   false,   true }},
        {"TMOD",  {false,  false,   false,   false}},
        {"SYNC",  {false,  false,   false,   false}},
        {"EOC",   {false,  false,   false,   false}},
        {"UNKNOWN",{false, false,   false,   false}},
      }
    );

    inline static constexpr ImplDef m_requests = {
      "read", "write", "all-bank-refresh", "per-bank-refresh"
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        {"read", "RD"}, {"write", "WR"}, {"all-bank-refresh", "REFab"}, {"per-bank-refresh", "REFsb"},
      }
    );

    // AiM request set — PCG-specific. Order MUST match the Opcode enum integer values in
    // base/request.h, because the controller indexes m_aim_request_translations by (int)req.opcode.
    inline static constexpr ImplDef m_aim_requests = {
      "MIN", "ISR_GGM_EXTEND", "ISR_GGM_REDUCE", "ISR_EOC", "ISR_SYNC",
      "ISR_NET_DELAY", "ISR_NET_WAIT", "MAX"
    };

    inline static const ImplLUT m_aim_request_translations = LUT (
      m_aim_requests, m_commands, {
        {"MIN", "UNKNOWN"},
        {"ISR_GGM_EXTEND", "RD"},   // load/store phases use RD/WR via the phase-tagged sub-requests
        {"ISR_GGM_REDUCE", "RD"},
        {"ISR_EOC", "EOC"}, {"ISR_SYNC", "SYNC"},
        {"ISR_NET_DELAY", "UNKNOWN"}, {"ISR_NET_WAIT", "UNKNOWN"},  // frontend-only, never issued
        {"MAX", "UNKNOWN"},
      }
    );

   
  /************************************************
   *                   Timing
   ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL", "nCL", "nRCDRD", "nRCDWR", "nRP", "nRAS", "nRC", "nWR", "nRTPS", "nRTPL", "nCWL",
      "nCCDS", "nCCDL",
      "nRRDS", "nRRDL",
      "nWTRS", "nWTRL",
      "nRTW",
      "nFAW",
      "nRFC", "nRFCSB", "nREFI", "nREFISB", "nRREFD",
      "tCK_ps"
    };


  /************************************************
   *                 Node States
   ***********************************************/
    inline static constexpr ImplDef m_states = {
       "Opened", "Closed", "N/A"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",       "N/A"}, 
        {"pseudochannel", "N/A"}, 
        {"bankgroup",     "N/A"},
        {"bank",          "Closed"},
        {"row",           "Closed"},
        {"column",        "N/A"},
      }
    );

  public:
    struct Node : public DRAMNodeBase<HBM2> {
      Node(HBM2* dram, Node* parent, int level, int id) : DRAMNodeBase<HBM2>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;


  public:
    void tick() override {
      m_clk++;
    };

    void init() override {
      RAMULATOR_DECLARE_SPECS();
      set_organization();
      set_timing_vals();

      set_actions();
      set_preqs();
      set_rowhits();
      set_rowopens();
      
      create_nodes();
    };

    void issue_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      m_channels[channel_id]->update_timing(command, addr_vec, m_clk);
      m_channels[channel_id]->update_states(command, addr_vec, m_clk);
      // Maintain m_open_rows[channel] as a count of currently-open banks (drives the
      // active/precharged cycle stat in the controller). ACT opens one bank; PRE/RDA/WRA
      // close one; PREA closes all in the channel.
      if (command == m_commands["ACT"]) {
        m_open_rows[channel_id] += 1;
      } else if (command == m_commands["PRE"] || command == m_commands["RDA"] || command == m_commands["WRA"]) {
        if (m_open_rows[channel_id] > 0) m_open_rows[channel_id] -= 1;
      } else if (command == m_commands["PREA"]) {
        m_open_rows[channel_id] = 0;
      } else if (command == m_commands["ACT16"]) {
        // All-bank activate opens every bank in the channel at once.
        m_open_rows[channel_id] = m_organization.count[m_levels["bankgroup"]]
                                * m_organization.count[m_levels["bank"]];
      }
    };

    int get_preq_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->get_preq_command(command, addr_vec, m_clk);
    };

    bool check_ready(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_ready(command, addr_vec, m_clk);
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_rowbuffer_hit(command, addr_vec, m_clk);
    };

  private:
    void set_organization() {
      // Channel width
      m_channel_width = param_group("org").param<int>("channel_width").default_val(64);

      // Organization
      m_organization.count.resize(m_levels.size(), -1);

      // Load organization preset if provided
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
        if (org_presets.count(*preset_name) > 0) {
          m_organization = org_presets.at(*preset_name);
        } else {
          throw ConfigurationError("Unrecognized organization preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Override the preset with any provided settings
      if (auto dq = param_group("org").param<int>("dq").optional()) {
        m_organization.dq = *dq;
      }

      for (int i = 0; i < m_levels.size(); i++){
        auto level_name = m_levels(i);
        if (auto sz = param_group("org").param<int>(level_name).optional()) {
          m_organization.count[i] = *sz;
        }
      }

      if (auto density = param_group("org").param<int>("density").optional()) {
        m_organization.density = *density;
      }

      // Sanity check: is the calculated channel density the same as the provided one?
      size_t _density = size_t(m_organization.count[m_levels["pseudochannel"]]) *
                        size_t(m_organization.count[m_levels["bankgroup"]]) *
                        size_t(m_organization.count[m_levels["bank"]]) *
                        size_t(m_organization.count[m_levels["row"]]) *
                        size_t(m_organization.count[m_levels["column"]]) *
                        size_t(m_organization.dq);
      _density >>= 20;
      if (m_organization.density != _density) {
        throw ConfigurationError(
            "Calculated {} channel density {} Mb does not equal the provided density {} Mb!", 
            get_name(),
            _density, 
            m_organization.density
        );
      }

    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);
      m_command_latencies.resize(m_commands.size(), -1);

      // Load timing preset if provided
      bool preset_provided = false;
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        if (timing_presets.count(*preset_name) > 0) {
          m_timing_vals = timing_presets.at(*preset_name);
          preset_provided = true;
        } else {
          throw ConfigurationError("Unrecognized timing preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Check for rate (in MT/s), and if provided, calculate and set tCK (in picosecond)
      if (auto dq = param_group("timing").param<int>("rate").optional()) {
        if (preset_provided) {
          throw ConfigurationError("Cannot change the transfer rate of {} when using a speed preset !", get_name());
        }
        m_timing_vals("rate") = *dq;
      }
      int tCK_ps = 1E6 / (m_timing_vals("rate") / 2);
      m_timing_vals("tCK_ps") = tCK_ps;

      // Refresh timings
      // tRFC table (unit is nanosecond!)
      constexpr int tRFC_TABLE[1][4] = {
      //  2Gb   4Gb   8Gb  16Gb
        { 160,  260,  350,  450},
      };

      // tRFC table (unit is nanosecond!)
      constexpr int tREFISB_TABLE[1][4] = {
      //  2Gb    4Gb    8Gb    16Gb
        { 4875,  4875,  2438,  2438},
      };

      int density_id = [](int density_Mb) -> int { 
        switch (density_Mb) {
          case 2048:  return 0;
          case 4096:  return 1;
          case 8192:  return 2;
          case 16384: return 3;
          default:    return -1;
        }
      }(m_organization.density);

      m_timing_vals("nRFC")  = JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);
      m_timing_vals("nREFISB")  = JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);

      // Overwrite timing parameters with any user-provided value
      // Rate and tCK should not be overwritten
      for (int i = 1; i < m_timings.size() - 1; i++) {
        auto timing_name = std::string(m_timings(i));

        if (auto provided_timing = param_group("timing").param<int>(timing_name).optional()) {
          // Check if the user specifies in the number of cycles (e.g., nRCD)
          m_timing_vals(i) = *provided_timing;
        } else if (auto provided_timing = param_group("timing").param<float>(timing_name.replace(0, 1, "t")).optional()) {
          // Check if the user specifies in nanoseconds (e.g., tRCD)
          m_timing_vals(i) = JEDEC_rounding(*provided_timing, tCK_ps);
        }
      }

      // Check if there is any uninitialized timings
      for (int i = 0; i < m_timing_vals.size(); i++) {
        if (m_timing_vals(i) == -1) {
          throw ConfigurationError("In \"{}\", timing {} is not specified!", get_name(), m_timings(i));
        }
      }      

      // Set read latency
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

      // Per-command latencies (used by the AiM controller for the final command's depart time).
      m_command_latencies("RD")  = m_timing_vals("nCL")  + m_timing_vals("nBL");
      m_command_latencies("RDA") = m_timing_vals("nCL")  + m_timing_vals("nBL");
      m_command_latencies("WR")  = m_timing_vals("nCWL") + m_timing_vals("nBL");
      m_command_latencies("WRA") = m_timing_vals("nCWL") + m_timing_vals("nBL");
      // All-bank PIM column commands mirror RD/WR; ACT16 is only ever a preq intermediate (never a
      // final_command), so its latency is never read for depart time — set >0 to satisfy assert.
      m_command_latencies("ABRD") = m_timing_vals("nCL")  + m_timing_vals("nBL");
      m_command_latencies("ABWR") = m_timing_vals("nCWL") + m_timing_vals("nBL");
      m_command_latencies("ACT16") = m_timing_vals("nRCDRD");
      m_command_latencies("REFab") = m_timing_vals("nRFC");
      m_command_latencies("REFsb") = m_timing_vals("nRFC");
      m_command_latencies("TMOD") = 1;
      m_command_latencies("SYNC") = 1;
      m_command_latencies("EOC")  = 1;
      m_command_latencies("UNKNOWN") = 1;

      // Populate the timing constraints
      #define V(timing) (m_timing_vals(timing))
      populate_timingcons(this, {
          /*** Channel ***/
          /// 2-cycle ACT command (for row commands)
          {.level = "channel", .preceding = {"ACT"}, .following = {"ACT", "PRE", "PREA", "REFab", "REFsb"}, .latency = 2},

          /*** All-bank PIM commands (channel scope — one bus event drives every bank in parallel) ***/
          // These mirror the single-bank ACT/RD/WR constraints but live at CHANNEL level so the ready-
          // clock is gated on the channel node. ACT16 is deliberately ABSENT from the nFAW window=4
          // constraint: one all-bank activate is a single command, not 4 separate row activations.
          // RAS <-> RAS / RAS <-> CAS (all-bank row cycle)
          {.level = "channel", .preceding = {"ACT16"}, .following = {"ACT16"}, .latency = V("nRC")},
          {.level = "channel", .preceding = {"ACT16"}, .following = {"ABRD"}, .latency = V("nRCDRD")},
          {.level = "channel", .preceding = {"ACT16"}, .following = {"ABWR"}, .latency = V("nRCDWR")},
          {.level = "channel", .preceding = {"ACT16"}, .following = {"PREA"}, .latency = V("nRAS")},
          {.level = "channel", .preceding = {"PREA"},  .following = {"ACT16"}, .latency = V("nRP")},
          // CAS <-> CAS (data-bus occupancy + column-to-column; use the same-bankgroup nCCDL since an
          // all-bank command logically contends across every bankgroup at once)
          {.level = "channel", .preceding = {"ABRD"}, .following = {"ABRD", "ABWR"}, .latency = V("nBL")},
          {.level = "channel", .preceding = {"ABWR"}, .following = {"ABRD", "ABWR"}, .latency = V("nBL")},
          {.level = "channel", .preceding = {"ABRD"}, .following = {"ABRD"}, .latency = V("nCCDL")},
          {.level = "channel", .preceding = {"ABWR"}, .following = {"ABWR"}, .latency = V("nCCDL")},
          {.level = "channel", .preceding = {"ABRD"}, .following = {"ABWR"}, .latency = V("nCL") + V("nBL") + 2 - V("nCWL")},
          {.level = "channel", .preceding = {"ABWR"}, .following = {"ABRD"}, .latency = V("nCWL") + V("nBL") + V("nWTRL")},
          // CAS <-> PREA (so the read->write row cycle is correctly spaced)
          {.level = "channel", .preceding = {"ABRD"}, .following = {"PREA"}, .latency = V("nRTPS")},
          {.level = "channel", .preceding = {"ABWR"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},
          // Refresh interlock (all-bank ACT cannot overlap a refresh, and vice-versa)
          {.level = "channel", .preceding = {"REFab"}, .following = {"ACT16"}, .latency = V("nRFC")},
          {.level = "channel", .preceding = {"ACT16"}, .following = {"REFab"}, .latency = V("nRC")},

          /*** Pseudo Channel (Table 3 — Array Access Timings Counted Individually Per Pseudo Channel, JESD-235C) ***/ 
          // RAS <-> RAS
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},
          /// 4-activation window restriction
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},

          /// ACT actually happens on the 2-nd cycle of ACT, so +1 cycle to nRRD
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"REFsb"}, .latency = V("nRRDS") + 1},
          /// nRREFD is the latency between REFsb <-> REFsb to *different* banks
          {.level = "pseudochannel", .preceding = {"REFsb"}, .following = {"REFsb"}, .latency = V("nRREFD")},
          /// nRREFD is the latency between REFsb <-> ACT to *different* banks. -1 as ACT happens on its 2nd cycle
          {.level = "pseudochannel", .preceding = {"REFsb"}, .following = {"ACT"}, .latency = V("nRREFD") - 1},

          // CAS <-> CAS
          /// Data bus occupancy
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nBL")},
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nBL")},

          // CAS <-> CAS
          /// nCCDS is the minimal latency for column commands 
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDS")},
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDS")},
          /// RD <-> WR, Minimum Read to Write, Assuming tWPRE = 1 tCK                          
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"WR", "WRA"}, .latency = V("nCL") + V("nBL") + 2 - V("nCWL")},
          /// WR <-> RD, Minimum Read after Write
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRS")},
          /// CAS <-> PREab
          {.level = "pseudochannel", .preceding = {"RD"}, .following = {"PREA"}, .latency = V("nRTPS")},
          {.level = "pseudochannel", .preceding = {"WR"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},          
          /// RAS <-> RAS
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},          
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},          
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"PREA"}, .latency = V("nRAS")},          
          {.level = "pseudochannel", .preceding = {"PREA"}, .following = {"ACT"}, .latency = V("nRP")},          
          /// RAS <-> REF
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"REFab"}, .latency = V("nRC")},          
          {.level = "pseudochannel", .preceding = {"PRE", "PREA"}, .following = {"REFab"}, .latency = V("nRP")},          
          {.level = "pseudochannel", .preceding = {"RDA"}, .following = {"REFab"}, .latency = V("nRP") + V("nRTPS")},          
          {.level = "pseudochannel", .preceding = {"WRA"}, .following = {"REFab"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},          
          {.level = "pseudochannel", .preceding = {"REFab"}, .following = {"ACT", "REFsb"}, .latency = V("nRFC")},          

          /*** Same Bank Group ***/ 
          /// CAS <-> CAS
          {.level = "bankgroup", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRL")},
          /// RAS <-> RAS
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDL")},  
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"REFsb"}, .latency = V("nRRDL") + 1},  
          {.level = "bankgroup", .preceding = {"REFsb"}, .following = {"ACT"}, .latency = V("nRRDL") - 1},  

          {.level = "bank", .preceding = {"RD"},  .following = {"PRE"}, .latency = V("nRTPS")},  


          /*** Bank ***/ 
          {.level = "bank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"RD", "RDA"}, .latency = V("nRCDRD")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"WR", "WRA"}, .latency = V("nRCDWR")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"PRE"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"PRE"}, .following = {"ACT"}, .latency = V("nRP")},  
          {.level = "bank", .preceding = {"RD"},  .following = {"PRE"}, .latency = V("nRTPL")},  
          {.level = "bank", .preceding = {"WR"},  .following = {"PRE"}, .latency = V("nCWL") + V("nBL") + V("nWR")},  
          {.level = "bank", .preceding = {"RDA"}, .following = {"ACT", "REFsb"}, .latency = V("nRTPL") + V("nRP")},  
          {.level = "bank", .preceding = {"WRA"}, .following = {"ACT", "REFsb"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},  
        }
      );
      #undef V

    };

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));

      // Channel Actions
      m_actions[m_levels["channel"]][m_commands["PREA"]] = Lambdas::Action::Channel::PREab<HBM2>;
      // All-bank activate opens the target row in EVERY bank (ABRD/ABWR change no state, like RD/WR).
      m_actions[m_levels["channel"]][m_commands["ACT16"]] = Lambdas::Action::Channel::ACTab<HBM2>;

      // Bank actions
      m_actions[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Action::Bank::ACT<HBM2>;
      m_actions[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Action::Bank::PRE<HBM2>;
      m_actions[m_levels["bank"]][m_commands["RDA"]] = Lambdas::Action::Bank::PRE<HBM2>;
      m_actions[m_levels["bank"]][m_commands["WRA"]] = Lambdas::Action::Bank::PRE<HBM2>;
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));

      // Channel Actions
      m_preqs[m_levels["channel"]][m_commands["REFab"]] = Lambdas::Preq::Channel::RequireAllBanksClosed<HBM2>;
      // All-bank PIM preqs: ABRD/ABWR need the target row open in ALL banks (else PREA / ACT16);
      // ACT16 needs all banks closed first (else PREA). Mirrors the REFab all-bank precedent.
      m_preqs[m_levels["channel"]][m_commands["ABRD"]] = Lambdas::Preq::Channel::RequireAllRowsOpen<HBM2>;
      m_preqs[m_levels["channel"]][m_commands["ABWR"]] = Lambdas::Preq::Channel::RequireAllRowsOpen<HBM2>;
      m_preqs[m_levels["channel"]][m_commands["ACT16"]] = Lambdas::Preq::Channel::RequireAllBanksClosed<HBM2>;

      // Bank actions
      m_preqs[m_levels["bank"]][m_commands["REFsb"]] = Lambdas::Preq::Bank::RequireBankClosed<HBM2>;
      m_preqs[m_levels["bank"]][m_commands["RD"]] = Lambdas::Preq::Bank::RequireRowOpen<HBM2>;
      m_preqs[m_levels["bank"]][m_commands["WR"]] = Lambdas::Preq::Bank::RequireRowOpen<HBM2>;
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowhits[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowHit::Bank::RDWR<HBM2>;
      m_rowhits[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowHit::Bank::RDWR<HBM2>;
    }


    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowopens[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowOpen::Bank::RDWR<HBM2>;
      m_rowopens[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowOpen::Bank::RDWR<HBM2>;
    }


    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        Node* channel = new Node(this, nullptr, 0, i);
        m_channels.push_back(channel);
        // m_open_rows[ch] is read every controller tick for the active/precharged stat.
        // Track it as a COUNT of currently-open banks (HBM2 has up to 32 banks/channel,
        // which overflows GDDR6's uint16_t bitmask approach), tested only against 0.
        m_open_rows.push_back(0);
      }
    };
};


}        // namespace Ramulator