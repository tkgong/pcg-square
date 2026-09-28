#include <vector>

#include "base/base.h"
#include "base/request.h"
#include "dram_controller/controller.h"
#include "dram_controller/refresh.h"

namespace Ramulator {

class AllBankRefresh : public IRefreshManager, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IRefreshManager, AllBankRefresh, "AllBank", "All-Bank Refresh scheme.")
private:
    Clk_t m_clk = 0;
    IDRAM *m_dram;
    IDRAMController *m_ctrl;

    int m_dram_org_levels = -1;
    int m_num_ranks = -1;

    int m_nrefi = -1;
    int m_ref_req_id = -1;
    Clk_t m_next_refresh_cycle = -1;

public:
    bool m_enable = true;

    void init() override {
        m_ctrl = cast_parent<IDRAMController>();
        m_enable = param<bool>("enable").desc("Issue all-bank refreshes every nREFI.").default_val(true);
    };

    void setup(IFrontEnd *frontend, IMemorySystem *memory_system) override {
        m_dram = m_ctrl->m_dram;

        m_dram_org_levels = m_dram->m_levels.size();
        m_num_ranks = m_dram->get_level_size("rank");
        // Organisations without a rank level (the HBM2 class, also used for the
        // GDDR6/GDDR7 presets) returned -1 here, so the loop below never ran and
        // no refresh was ever issued. Refresh every pseudochannel (or the whole
        // channel) instead.
        if (m_num_ranks < 0) {
            const int npc = m_dram->get_level_size("pseudochannel");
            m_num_ranks = npc > 0 ? npc : 1;
        }
        if (!m_enable) m_num_ranks = 0;

        m_nrefi = m_dram->m_timing_vals("nREFI");
        m_ref_req_id = m_dram->m_requests("all-bank-refresh");

        m_next_refresh_cycle = m_nrefi;
    };

    void tick() override {
        m_clk++;

        if (m_clk == m_next_refresh_cycle) {
            m_next_refresh_cycle += m_nrefi;
            for (int r = 0; r < m_num_ranks; r++) {
                std::vector<int> addr_vec(m_dram_org_levels, -1);
                addr_vec[0] = m_ctrl->m_channel_id;
                addr_vec[1] = r;
                Request req(addr_vec, m_ref_req_id);

                bool is_success = m_ctrl->priority_send(req);
                if (!is_success) {
                    throw std::runtime_error("Failed to send refresh!");
                }
            }
        }
    };
};

} // namespace Ramulator
