// SPU-side blocks, after PIM/docs/pim_isa_assembly.md: XADD array (8 x 32-bit add, with the mod-ALU hooks: 64-bit carry
// link, 62-bit mask and sparse-prime fold, predicated write), VXORL array (8 x 32-bit xor + rotate), the ChaCha8 EXTEND
// micro-sequencer (dual issue, quarter-round pairs in anti-phase, 9 ticks per pair), load skid, store queue + PACK.
`default_nettype none

(* keep_hierarchy *)
module xadd(input wire [255:0] a, input wire [255:0] b, input wire link, input wire mask62,
            output wire [255:0] y, output wire [3:0] cout);
  genvar g;
  generate for (g = 0; g < 4; g = g + 1) begin : P
    wire [31:0] alo = a[64*g +: 32];
    wire [31:0] ahi = mask62 ? {2'b00, a[64*g + 32 +: 30]} : a[64*g + 32 +: 32];
    wire [32:0] lo = {1'b0, alo} + {1'b0, b[64*g +: 32]};
    wire [32:0] hi = {1'b0, ahi} + {1'b0, b[64*g + 32 +: 32]} + {32'd0, link & lo[32]};
    assign y[64*g +: 64] = {hi[31:0], lo[31:0]};
    assign cout[g] = hi[32];
  end endgenerate
endmodule

// fold constant of the sparse prime P = 2^62 - 3*2^25 + 1: 2^62 = 3*2^25 - 1 (mod P); hi = top two bits of each 64-bit lane
(* keep_hierarchy *)
module foldk(input wire [255:0] a, output wire [255:0] k);
  localparam [63:0] K = 64'd100663295;
  genvar g;
  generate for (g = 0; g < 4; g = g + 1) begin : P
    wire [1:0] h = a[64*g + 62 +: 2];
    assign k[64*g +: 64] = (h == 2'd0) ? 64'd0 : (h == 2'd1) ? K : (h == 2'd2) ? (K << 1) : (K + (K << 1));
  end endgenerate
endmodule

(* keep_hierarchy *)
module vxorl(input wire [255:0] a, input wire [255:0] b, input wire [2:0] rot, output wire [255:0] y);
  genvar g;
  generate for (g = 0; g < 8; g = g + 1) begin : L
    wire [31:0] x = a[32*g +: 32] ^ b[32*g +: 32];
    assign y[32*g +: 32] = (rot == 3'd1) ? {x[15:0], x[31:16]} : (rot == 3'd2) ? {x[19:0], x[31:20]} :
                           (rot == 3'd3) ? {x[23:0], x[31:24]} : (rot == 3'd4) ? {x[24:0], x[31:25]} : x;
  end endgenerate
endmodule

// EXTEND micro-sequencer.  144 core ticks + 4 feed-forward ticks.  Registers before their first write in a batch are
// read through the constant / skid bypass, so no prologue writes are needed.
(* keep_hierarchy *)
module useq(
  input wire clk, input wire rst, input wire go, output wire busy,
  output reg xa_v, output reg [2:0] xa_ra, output reg [2:0] xa_rb, output reg xa_op1c, output reg xa_op2c,
  output reg [4:0] xa_s1, output reg [4:0] xa_s2, output reg xa_tosq, output reg [1:0] xa_sq, output reg [2:0] xa_wa,
  output reg xr_v, output reg [2:0] xr_rb, output reg [2:0] xr_ra, output reg [1:0] xr_op1s, output reg [4:0] xr_s1,
  output reg [1:0] xr_sk, output reg [2:0] xr_rot, output reg [2:0] xr_wb);
  reg run; reg [1:0] dr; reg half; reg pair; reg [3:0] tick; reg epi;
  assign busy = run;
  wire last_tick = epi ? (tick == 4'd3) : (tick == 4'd8);
  always @(posedge clk) begin
    if (rst) begin run <= 1'b0; dr <= 0; half <= 0; pair <= 0; tick <= 0; epi <= 0; end
    else if (!run) begin if (go) begin run <= 1'b1; dr <= 0; half <= 0; pair <= 0; tick <= 0; epi <= 0; end end
    else if (!last_tick) tick <= tick + 4'd1;
    else begin
      tick <= 4'd0;
      if (epi) run <= 1'b0;
      else begin
        pair <= ~pair;
        if (pair) begin half <= ~half; if (half) begin dr <= dr + 2'd1; if (dr == 2'd3) epi <= 1'b1; end end
      end
    end
  end
  // quarter-round i (0..3) of the current half: a = i, b = 4 + (i + half), c = 8 + (i + 2 half), d = 12 + (i + 3 half)
  wire first = (dr == 2'd0) & ~half;
  wire [3:0] sA = tick, sB = tick - 4'd1;
  wire xaA = ~tick[0] & (tick != 4'd8);             // XADD serves chain A on even ticks 0..6, chain B on odd ticks
  wire [1:0] xi = {pair, ~xaA};
  wire [3:0] xs = xaA ? sA : sB;
  wire xrA = tick[0];                               // VXORL serves chain A on odd ticks, chain B on even ticks 2..8
  wire [1:0] ri = {pair, ~xrA};
  wire [3:0] rs = xrA ? sA : sB;
  wire [1:0] xb = xi + {1'b0, half}, xc = xi + {half, 1'b0}, xd = xi + {half, half};
  wire [1:0] rb = ri + {1'b0, half}, rc = ri + {half, 1'b0}, rd = ri + {half, half};
  always @* begin
    xa_v = 0; xa_ra = 0; xa_rb = 0; xa_op1c = 0; xa_op2c = 0; xa_s1 = 0; xa_s2 = 0; xa_tosq = 0; xa_sq = 0; xa_wa = 0;
    xr_v = 0; xr_rb = 0; xr_ra = 0; xr_op1s = 0; xr_s1 = 0; xr_sk = 0; xr_rot = 0; xr_wb = 0;
    if (run & epi) begin                            // feed-forward of words 0..3 into the store queue
      xa_v = 1; xa_ra = {1'b0, tick[1:0]}; xa_op2c = 1; xa_s2 = {3'd1, tick[1:0]}; xa_tosq = 1; xa_sq = tick[1:0];
    end else if (run) begin
      if (xaA | tick[0]) begin
        xa_v = 1;
        if (xs[1] == 1'b0) begin                    // a += b
          xa_ra = {1'b0, xi}; xa_rb = {1'b0, xb}; xa_wa = {1'b0, xi};
          if (first & (xs == 4'd0)) begin xa_op1c = 1; xa_s1 = {3'd0, xi}; xa_op2c = 1; xa_s2 = {3'd0, xb}; end
        end else begin                              // c += d
          xa_ra = {1'b1, xc}; xa_rb = {1'b1, xd}; xa_wa = {1'b1, xc};
          if (first & (xs == 4'd2)) begin xa_op1c = 1; xa_s1 = {3'd1, xc}; end
        end
      end
      if (xrA | (tick >= 4'd2)) begin
        xr_v = 1;
        if (rs[1] == 1'b0) begin                    // d = (d ^ a) <<< 16 or 8
          xr_rb = {1'b1, rd}; xr_ra = {1'b0, ri}; xr_wb = {1'b1, rd}; xr_rot = (rs == 4'd1) ? 3'd1 : 3'd3;
          if (first & (rs == 4'd1)) begin xr_op1s = 2'd2; xr_sk = rd; end
        end else begin                              // b = (b ^ c) <<< 12 or 7
          xr_rb = {1'b0, rb}; xr_ra = {1'b1, rc}; xr_wb = {1'b0, rb}; xr_rot = (rs == 4'd3) ? 3'd2 : 3'd4;
          if (first & (rs == 4'd3)) begin xr_op1s = 2'd1; xr_s1 = {3'd0, rb}; end
        end
      end
    end
  end
endmodule

// Load skid: four columns (the next batch's seeds; read once through the bypass, then free).  Store queue: four columns
// of feed-forward results.  On the way out each column pair gets the correction word, the free child (xor with the
// parent, re-read from the still-open row and presented on din, as in the documented REREAD scheme) and the
// cross-lane child interleave.
(* keep_hierarchy *)
module lsu(
  input wire clk, input wire rst, input wire go, input wire [255:0] din, input wire sk_push,
  input wire [1:0] sk_ra, output wire [255:0] sk_q,
  input wire sq_we, input wire [1:0] sq_wa, input wire [255:0] sq_wd,
  input wire sq_pop, input wire [127:0] cw, output wire [255:0] st_col);
  reg [255:0] sk [0:3]; reg [255:0] sq [0:3];
  reg [1:0] wp; reg [2:0] rp; reg [7:0] tcur, tdr;
  integer i;
  always @(posedge clk) begin
    if (rst) begin wp <= 2'd0; rp <= 3'd0; end
    else begin
      if (go) wp <= 2'd0; else if (sk_push) wp <= wp + 2'd1;
      if (sq_pop) rp <= rp + 3'd1;
    end
    if (sk_push) sk[wp] <= din;
    if (sq_we) sq[sq_wa] <= sq_wd;
    if (go) for (i = 0; i < 8; i = i + 1) tcur[i] <= sk[0][32*i];        // control bits of the batch in flight
    if (sq_we & (sq_wa == 2'd0)) tdr <= tcur;                             // ... and of the batch being drained
  end
  assign sk_q = sk[sk_ra];
  wire [255:0] lf = sq[rp[2:1]];
  wire [31:0] cww = cw[32*rp[2:1] +: 32];
  genvar g;
  generate for (g = 0; g < 4; g = g + 1) begin : I
    wire [2:0] ln = {rp[0], g[1:0]};
    wire [31:0] l = lf[32*ln +: 32] ^ (cww & {32{tdr[ln]}});
    assign st_col[64*g +: 32] = l;
    assign st_col[64*g + 32 +: 32] = l ^ din[32*ln +: 32];
  end endgenerate
endmodule
