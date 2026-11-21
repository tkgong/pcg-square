// Processing unit, three configurations of one source:
//   BF16=1 ARX=0  AiM-function unit (BF16 MAC / element-wise multiply / element-wise add) on an HBM-PIM-style frame
//                 (two register groups of 8 x 256 b, scalar registers, 32-entry instruction memory, column-stepped)
//   BF16=0 ARX=1  the SPU (PIM/docs/pim_isa_assembly.md)
//   BF16=1 ARX=1  the unified unit: one register file, one instruction memory and decoder, one bank data path
`default_nettype none

(* keep_hierarchy *)
module rf #(parameter ARX = 1)(
  input wire clk,
  input wire [2:0] a_r0, input wire [2:0] b_r0, input wire [2:0] b_r1,
  output wire [255:0] a_q0, output wire [255:0] b_q0, output wire [255:0] b_q1,
  input wire a_we, input wire [2:0] a_wa, input wire [3:0] a_wl, input wire [255:0] a_wd,
  input wire b_we, input wire [2:0] b_wa, input wire [3:0] b_wl, input wire [255:0] b_wd);
  reg [63:0] ga [0:31]; reg [63:0] gb [0:31];          // entry e, 64-bit lane l at index 4 e + l
  genvar g;
  generate for (g = 0; g < 4; g = g + 1) begin : W
    wire [1:0] ln = g[1:0];
    always @(posedge clk) begin
      if (a_we & a_wl[g]) ga[{a_wa, ln}] <= a_wd[64*g +: 64];
      if (b_we & b_wl[g]) gb[{b_wa, ln}] <= b_wd[64*g +: 64];
    end
    assign a_q0[64*g +: 64] = ga[{a_r0, ln}];
    assign b_q0[64*g +: 64] = gb[{b_r0, ln}];
    if (ARX) begin : R1
      assign b_q1[64*g +: 64] = gb[{b_r1, ln}];
    end else begin : R0
      assign b_q1[64*g +: 64] = 64'd0;
    end
  end endgenerate
endmodule

// scalar registers: 16 x 16 b (BF16 scalars).  With ARX the same 256 bits hold the eight ChaCha key-schedule words
// (word w = entries 2w+1, 2w), read as 32-bit words on three small ports; four more 32-bit registers hold the
// correction word.  The ChaCha sigma words and the mod-ALU constants are hard-wired.
(* keep_hierarchy *)
module srf #(parameter ARX = 1)(
  input wire clk, input wire we, input wire [4:0] wa, input wire [31:0] wd,
  input wire [3:0] r_bf, output wire [15:0] q_bf,
  input wire [1:0] kb0, input wire [1:0] kb1, input wire [1:0] kc,
  output wire [31:0] q_kb0, output wire [31:0] q_kb1, output wire [31:0] q_kc, output wire [127:0] cw);
  reg [15:0] m [0:15];
  always @(posedge clk) if (we & ~wa[4]) m[wa[3:0]] <= wd[15:0];
  assign q_bf = m[r_bf];
  generate if (ARX) begin : A
    reg [31:0] c [0:3];
    always @(posedge clk) if (we & (wa[4:2] == 3'b100)) c[wa[1:0]] <= wd;
    assign q_kb0 = {m[{1'b0, kb0, 1'b1}], m[{1'b0, kb0, 1'b0}]};
    assign q_kb1 = {m[{1'b0, kb1, 1'b1}], m[{1'b0, kb1, 1'b0}]};
    assign q_kc = {m[{1'b1, kc, 1'b1}], m[{1'b1, kc, 1'b0}]};
    assign cw = {c[3], c[2], c[1], c[0]};
  end else begin : B
    assign q_kb0 = 32'd0; assign q_kb1 = 32'd0; assign q_kc = 32'd0; assign cw = 128'd0;
  end endgenerate
endmodule

// instruction memory (32 x 32 b), program counter, loop counter, decoder
(* keep_hierarchy *)
module ctrl #(parameter BF16 = 1, parameter ARX = 1)(
  input wire clk, input wire rst, input wire crf_we, input wire [4:0] crf_wa, input wire [31:0] crf_wd,
  input wire step, input wire pc_rst, input wire hold, output wire [31:0] ir, output wire ex);
  reg [31:0] crf [0:31]; reg [4:0] pc; reg [9:0] lc; reg lca;
  always @(posedge clk) if (crf_we) crf[crf_wa] <= crf_wd;
  assign ir = crf[pc];
  assign ex = step & ~hold;
  wire jmp = ex & (ir[31:27] == 5'd8);
  wire take = jmp & ~(lca & (lc == 10'd1));
  always @(posedge clk) begin
    if (rst | pc_rst) begin pc <= 5'd0; lc <= 10'd0; lca <= 1'b0; end
    else if (ex) begin
      pc <= (ir[31:27] == 5'd9) ? 5'd0 : take ? ir[26:22] : pc + 5'd1;
      if (jmp) begin
        if (~lca) begin lc <= ir[21:12]; lca <= 1'b1; end
        else if (lc == 10'd1) lca <= 1'b0;
        else lc <= lc - 10'd1;
      end
    end
  end
endmodule

module pu #(parameter BF16 = 1, parameter ARX = 1)(
  input wire clk, input wire rst,
  input wire crf_we, input wire [4:0] crf_wa, input wire [31:0] crf_wd,
  input wire srf_we, input wire [4:0] srf_wa, input wire [31:0] srf_wd,
  input wire step, input wire pc_rst,
  input wire [255:0] din, input wire sk_push, input wire sq_pop,
  output wire [255:0] dout, output wire busy);

  wire [31:0] ir; wire ex;
  ctrl #(.BF16(BF16), .ARX(ARX)) u_ctrl(.clk(clk), .rst(rst), .crf_we(crf_we), .crf_wa(crf_wa), .crf_wd(crf_wd),
    .step(step), .pc_rst(pc_rst), .hold(busy), .ir(ir), .ex(ex));
  wire [4:0] opc = ir[31:27];
  wire dg = ir[26]; wire [2:0] da = ir[25:23], sa = ir[22:20], sb = ir[19:17]; wire f16 = ir[16];
  wire [4:0] sidx = ir[15:11]; wire [2:0] f_rot = ir[10:8];
  wire f7 = ir[7], f_fold = ir[6], f_link = ir[5], f_mask = ir[4], f_pg = ir[3], f_c64 = ir[2], f_sp = ir[1], f0 = ir[0];
  wire i_ld = ex & (opc == 5'd1), i_st = ex & (opc == 5'd2);
  wire i_mul = ex & (opc == 5'd3) & (BF16 != 0), i_add = ex & (opc == 5'd4) & (BF16 != 0);
  wire i_mac = ex & (opc == 5'd5) & (BF16 != 0), i_rdm = ex & (opc == 5'd6) & (BF16 != 0);
  wire i_clr = ex & (opc == 5'd7) & (BF16 != 0);
  wire i_va = ex & (opc == 5'd16) & (ARX != 0), i_vx = ex & (opc == 5'd17) & (ARX != 0);
  wire go = ex & (opc == 5'd18) & (ARX != 0);

  // ---- micro-sequencer (ARX)
  wire u_xa_v, u_xa_op1c, u_xa_op2c, u_xa_tosq, u_xr_v;
  wire [2:0] u_xa_ra, u_xa_rb, u_xa_wa, u_xr_rb, u_xr_ra, u_xr_rot, u_xr_wb;
  wire [4:0] u_xa_s1, u_xa_s2, u_xr_s1; wire [1:0] u_xa_sq, u_xr_op1s, u_xr_sk;
  generate if (ARX) begin : SEQ
    useq u(.clk(clk), .rst(rst), .go(go), .busy(busy),
      .xa_v(u_xa_v), .xa_ra(u_xa_ra), .xa_rb(u_xa_rb), .xa_op1c(u_xa_op1c), .xa_op2c(u_xa_op2c), .xa_s1(u_xa_s1),
      .xa_s2(u_xa_s2), .xa_tosq(u_xa_tosq), .xa_sq(u_xa_sq), .xa_wa(u_xa_wa),
      .xr_v(u_xr_v), .xr_rb(u_xr_rb), .xr_ra(u_xr_ra), .xr_op1s(u_xr_op1s), .xr_s1(u_xr_s1), .xr_sk(u_xr_sk),
      .xr_rot(u_xr_rot), .xr_wb(u_xr_wb));
  end else begin : NOSEQ
    assign busy = 1'b0; assign u_xa_v = 0; assign u_xa_op1c = 0; assign u_xa_op2c = 0; assign u_xa_tosq = 0;
    assign u_xr_v = 0; assign u_xa_ra = 0; assign u_xa_rb = 0; assign u_xa_wa = 0; assign u_xr_rb = 0; assign u_xr_ra = 0;
    assign u_xr_rot = 0; assign u_xr_wb = 0; assign u_xa_s1 = 0; assign u_xa_s2 = 0; assign u_xr_s1 = 0;
    assign u_xa_sq = 0; assign u_xr_op1s = 0; assign u_xr_sk = 0;
  end endgenerate

  // ---- merged control: the sequencer while busy, the decoder otherwise
  wire xa_v = busy ? u_xa_v : i_va;
  wire [2:0] xa_ra = busy ? u_xa_ra : sa, xa_rb = busy ? u_xa_rb : sb;      // single-step xor-rotate reads group A on r0
  wire xa_op1c = busy ? u_xa_op1c : f16, xa_op2c = busy ? u_xa_op2c : f7;
  wire [4:0] xa_s1 = busy ? u_xa_s1 : sidx, xa_s2 = busy ? u_xa_s2 : sidx;
  wire xa_fold = ~busy & f_fold, xa_link = ~busy & f_link, xa_mask = ~busy & f_mask, xa_pg = ~busy & f_pg;
  wire xa_c64 = ~busy & f_c64, xa_sp = ~busy & f_sp;
  wire xa_tosq = busy & u_xa_tosq; wire xa_dg = ~busy & dg; wire [2:0] xa_wa = busy ? u_xa_wa : da;
  wire xr_v = busy ? u_xr_v : i_vx;
  wire [2:0] xr_rb = busy ? u_xr_rb : sb;
  wire [1:0] xr_op1s = busy ? u_xr_op1s : {1'b0, f0}; wire xr_op2c = ~busy & f7;
  wire [4:0] xr_s1 = busy ? u_xr_s1 : sidx; wire [2:0] xr_rot = busy ? u_xr_rot : f_rot;
  wire [2:0] xr_wb = busy ? u_xr_wb : da;

  // ---- storage
  wire [255:0] a_q0, b_q0, b_q1, a_wd, b_wd; wire a_we, b_we; wire [2:0] a_wa, b_wa; wire [3:0] a_wl, b_wl;
  rf #(.ARX(ARX)) u_rf(.clk(clk), .a_r0(xa_ra), .b_r0(xa_rb), .b_r1(xr_rb), .a_q0(a_q0), .b_q0(b_q0), .b_q1(b_q1),
    .a_we(a_we), .a_wa(a_wa), .a_wl(a_wl), .a_wd(a_wd), .b_we(b_we), .b_wa(b_wa), .b_wl(b_wl), .b_wd(b_wd));
  wire [15:0] q_bf; wire [31:0] q_kb0, q_kb1, q_kc; wire [127:0] cw;
  srf #(.ARX(ARX)) u_srf(.clk(clk), .we(srf_we), .wa(srf_wa), .wd(srf_wd), .r_bf(sidx[3:0]), .q_bf(q_bf),
    .kb0(xa_s2[1:0]), .kb1(xr_s1[1:0]), .kc(xa_s1[1:0]), .q_kb0(q_kb0), .q_kb1(q_kb1), .q_kc(q_kc), .cw(cw));

  // ---- BF16
  wire [255:0] ewmul, ewadd; wire [15:0] mac_rd;
  generate if (BF16) begin : FP
    wire [255:0] opa = f16 ? din : a_q0;
    wire [255:0] opb = f7 ? {16{q_bf}} : b_q0;
    bf16_dp u(.clk(clk), .rst(rst), .a(opa), .b(opb), .mac_v(i_mac), .mac_clr(i_clr),
      .ewmul(ewmul), .ewadd(ewadd), .mac_rd(mac_rd));
  end else begin : NOFP
    assign ewmul = 256'd0; assign ewadd = 256'd0; assign mac_rd = 16'd0;
  end endgenerate
  wire [255:0] wbus = i_ld ? din : i_mul ? ewmul : i_add ? ewadd : {240'd0, mac_rd};
  wire cw_we = i_ld | i_mul | i_add | i_rdm;

  // ---- ARX
  wire [255:0] xy, ry, sk_q, st_col; wire [3:0] xc;
  generate if (ARX) begin : AX
    function [31:0] sigma(input [1:0] k);
      sigma = (k == 2'd0) ? 32'h61707865 : (k == 2'd1) ? 32'h3320646e : (k == 2'd2) ? 32'h79622d32 : 32'h6b206574;
    endfunction
    localparam [63:0] MINUSP = 64'hC000000005FFFFFF, PPLUS1 = 64'h3FFFFFFFFA000002;     // P = 2^62 - 3*2^25 + 1
    wire [255:0] c1 = (xa_s1[4:2] == 3'd0) ? {8{sigma(xa_s1[1:0])}} : (xa_s1[4:2] == 3'd1) ? {8{q_kc}} : {4{PPLUS1}};
    wire [255:0] c2 = (xa_s2[4:2] == 3'd0) ? {8{q_kb0}} : (xa_s2[4:2] == 3'd1) ? {8{sigma(xa_s2[1:0])}} :
                      (xa_s2[4:2] == 3'd2) ? {4{MINUSP}} : {4{cw[63:0]}};
    wire [255:0] pm = {{64{b_q0[192]}}, {64{b_q0[128]}}, {64{b_q0[64]}}, {64{b_q0[0]}}};
    wire [255:0] fk;
    wire [255:0] x1 = xa_op1c ? c1 : a_q0;
    foldk u_fk(.a(x1), .k(fk));
    wire [255:0] x2 = xa_fold ? fk : xa_op2c ? (xa_pg ? (c2 & pm) : c2) : b_q0;
    xadd u_xa(.a(x1), .b(x2), .link(xa_link), .mask62(xa_mask), .y(xy), .cout(xc));
    reg [255:0] fwd;                                   // the adder's last result: the xor-rotate's second operand
    always @(posedge clk) if (xa_v & busy) fwd <= xy;
    wire [255:0] r1 = (xr_op1s == 2'd2) ? sk_q : (xr_op1s == 2'd1) ? {8{q_kb1}} : b_q1;
    wire [255:0] r2 = busy ? fwd : xr_op2c ? {256{1'b1}} : a_q0;
    vxorl u_xr(.a(r1), .b(r2), .rot(xr_rot), .y(ry));
    lsu u_lsu(.clk(clk), .rst(rst), .go(go), .din(din), .sk_push(sk_push), .sk_ra(u_xr_sk), .sk_q(sk_q),
      .sq_we(xa_v & xa_tosq), .sq_wa(u_xa_sq), .sq_wd(xy), .sq_pop(sq_pop), .cw(cw), .st_col(st_col));
  end else begin : NOAX
    assign xy = 256'd0; assign ry = 256'd0; assign sk_q = 256'd0; assign st_col = 256'd0; assign xc = 4'd0;
  end endgenerate

  // ---- write ports: group A takes the adder or the common bus, group B the xor-rotate, the adder or the common bus
  wire xa_wa_g = xa_v & ~xa_tosq & ~xa_dg, xa_wb_g = xa_v & ~xa_tosq & xa_dg;
  assign a_we = xa_wa_g | (cw_we & ~dg);
  assign a_wa = xa_wa_g ? xa_wa : da;
  assign a_wd = xa_wa_g ? xy : wbus;
  assign a_wl = (xa_wa_g & xa_sp) ? xc : 4'hf;
  assign b_we = xr_v | xa_wb_g | (cw_we & dg);
  assign b_wa = xr_v ? xr_wb : xa_wb_g ? xa_wa : da;
  assign b_wd = xr_v ? ry : xa_wb_g ? xy : wbus;
  assign b_wl = (xa_wb_g & xa_sp) ? xc : 4'hf;

  // ---- bank write data latch
  reg [255:0] dout_r;
  always @(posedge clk) begin
    if (i_st) dout_r <= dg ? b_q0 : a_q0;
    else if (sq_pop & (ARX != 0)) dout_r <= st_col;
  end
  assign dout = dout_r;
endmodule

module pu_aim(input wire clk, input wire rst, input wire crf_we, input wire [4:0] crf_wa, input wire [31:0] crf_wd,
  input wire srf_we, input wire [4:0] srf_wa, input wire [31:0] srf_wd, input wire step, input wire pc_rst,
  input wire [255:0] din, input wire sk_push, input wire sq_pop, output wire [255:0] dout, output wire busy);
  pu #(.BF16(1), .ARX(0)) u(.clk(clk), .rst(rst), .crf_we(crf_we), .crf_wa(crf_wa), .crf_wd(crf_wd), .srf_we(srf_we),
    .srf_wa(srf_wa), .srf_wd(srf_wd), .step(step), .pc_rst(pc_rst), .din(din), .sk_push(sk_push), .sq_pop(sq_pop),
    .dout(dout), .busy(busy));
endmodule
module pu_spu(input wire clk, input wire rst, input wire crf_we, input wire [4:0] crf_wa, input wire [31:0] crf_wd,
  input wire srf_we, input wire [4:0] srf_wa, input wire [31:0] srf_wd, input wire step, input wire pc_rst,
  input wire [255:0] din, input wire sk_push, input wire sq_pop, output wire [255:0] dout, output wire busy);
  pu #(.BF16(0), .ARX(1)) u(.clk(clk), .rst(rst), .crf_we(crf_we), .crf_wa(crf_wa), .crf_wd(crf_wd), .srf_we(srf_we),
    .srf_wa(srf_wa), .srf_wd(srf_wd), .step(step), .pc_rst(pc_rst), .din(din), .sk_push(sk_push), .sq_pop(sq_pop),
    .dout(dout), .busy(busy));
endmodule
module pu_uni(input wire clk, input wire rst, input wire crf_we, input wire [4:0] crf_wa, input wire [31:0] crf_wd,
  input wire srf_we, input wire [4:0] srf_wa, input wire [31:0] srf_wd, input wire step, input wire pc_rst,
  input wire [255:0] din, input wire sk_push, input wire sq_pop, output wire [255:0] dout, output wire busy);
  pu #(.BF16(1), .ARX(1)) u(.clk(clk), .rst(rst), .crf_we(crf_we), .crf_wa(crf_wa), .crf_wd(crf_wd), .srf_we(srf_we),
    .srf_wa(srf_wa), .srf_wd(srf_wd), .step(step), .pc_rst(pc_rst), .din(din), .sk_push(sk_push), .sq_pop(sq_pop),
    .dout(dout), .busy(busy));
endmodule
