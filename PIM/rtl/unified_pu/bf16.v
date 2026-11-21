// BF16 datapath with AiM's function set: MAC (16 products summed into one accumulator per unit), element-wise
// multiply, element-wise add.  A reconstruction from the public description (16 BF16 multipliers and an adder tree
// per processing unit); not SK hynix's netlist.  Denormals flush to zero, round half up.
`default_nettype none

module bf16_mulcore(input wire [15:0] a, input wire [15:0] b, output wire s, output wire [8:0] e, output wire [15:0] m);
  wire z = (a[14:7] == 8'd0) | (b[14:7] == 8'd0);
  assign s = a[15] ^ b[15];
  assign e = z ? 9'd0 : ({1'b0, a[14:7]} + {1'b0, b[14:7]});          // biased by 254
  assign m = z ? 16'd0 : ({1'b1, a[6:0]} * {1'b1, b[6:0]});            // value = m * 2^(e - 254 - 14)
endmodule

module bf16_pack(input wire s, input wire signed [10:0] e, input wire [7:0] m, input wire r, input wire z, output wire [15:0] y);
  // m = 1.fffffff (8 bits), r = round bit, e = biased exponent before rounding
  wire [8:0] mr = {1'b0, m} + {8'd0, r};
  wire signed [10:0] e2 = e + {10'd0, mr[8]};
  wire [6:0] f = mr[8] ? 7'd0 : mr[6:0];
  wire under = z | (e2 <= 11'sd0);
  wire over = ~under & (e2 >= 11'sd255);
  assign y = under ? {s, 15'd0} : over ? {s, 8'hff, 7'd0} : {s, e2[7:0], f};
endmodule

module bf16_ewmul(input wire s, input wire [8:0] e, input wire [15:0] m, output wire [15:0] y);
  wire hi = m[15];
  wire [7:0] mm = hi ? m[15:8] : m[14:7];
  wire r = hi ? m[7] : m[6];
  wire signed [10:0] eb = $signed({2'b0, e}) - 11'sd127 + {10'd0, hi};
  bf16_pack p(.s(s), .e(eb), .m(mm), .r(r), .z(m == 16'd0), .y(y));
endmodule

module bf16_add(input wire [15:0] a, input wire [15:0] b, output wire [15:0] y);
  wire ge = a[14:0] >= b[14:0];
  wire [15:0] x = ge ? a : b, w = ge ? b : a;
  wire [7:0] ex = x[14:7], ew = w[14:7];
  wire [10:0] mx = (ex == 0) ? 11'd0 : {1'b1, x[6:0], 3'b0};
  wire [10:0] mw0 = (ew == 0) ? 11'd0 : {1'b1, w[6:0], 3'b0};
  wire [7:0] d = ex - ew;
  wire [10:0] mw = (d >= 8'd11) ? 11'd0 : (mw0 >> d[3:0]);
  wire sub = x[15] ^ w[15];
  wire [11:0] sum = sub ? ({1'b0, mx} - {1'b0, mw}) : ({1'b0, mx} + {1'b0, mw});
  reg [3:0] lz; integer i;
  always @* begin lz = 4'd11; for (i = 0; i <= 10; i = i + 1) if (sum[i]) lz = 4'd10 - i[3:0]; end
  wire [10:0] sh = sum[10:0] << lz;
  wire [7:0] mm = sum[11] ? sum[11:4] : sh[10:3];
  wire r = sum[11] ? sum[3] : sh[2];
  wire signed [10:0] e = sum[11] ? ($signed({3'b0, ex}) + 11'sd1) : ($signed({3'b0, ex}) - $signed({7'b0, lz}));
  bf16_pack p(.s(x[15]), .e(e), .m(mm), .r(r), .z(sum == 12'd0), .y(y));
endmodule

// MAC stages 2 and 3 and the read-out: largest exponent (balanced tree), align, signed sum (balanced tree) | accumulate
(* keep_hierarchy *)
module bf16_acc(
  input wire clk, input wire rst, input wire v1, input wire [15:0] s1, input wire [143:0] e1, input wire [255:0] m1,
  input wire mac_clr, output wire [15:0] mac_rd);
  wire [9*31-1:0] et; wire [29*31-1:0] st;             // heaps: node n in slot n - 1, leaves 16..31
  genvar g;
  generate
    for (g = 15; g >= 1; g = g - 1) begin : EN
      wire [8:0] l = et[9*(2*g-1) +: 9], r = et[9*(2*g) +: 9];
      assign et[9*(g-1) +: 9] = (l > r) ? l : r;
    end
    for (g = 15; g >= 1; g = g - 1) begin : SN
      assign st[29*(g-1) +: 29] = $signed(st[29*(2*g-1) +: 29]) + $signed(st[29*(2*g) +: 29]);
    end
    for (g = 0; g < 16; g = g + 1) begin : LF
      assign et[9*(15+g) +: 9] = e1[9*g +: 9];
      wire [8:0] sh = et[8:0] - e1[9*g +: 9];
      wire [23:0] al = (sh >= 9'd24) ? 24'd0 : ({m1[16*g +: 16], 8'd0} >> sh[4:0]);
      assign st[29*(15+g) +: 29] = s1[g] ? -$signed({5'd0, al}) : $signed({5'd0, al});
    end
  endgenerate
  wire [8:0] emax = et[8:0];
  wire signed [28:0] tree = st[28:0];
  reg signed [28:0] sum2; reg [8:0] e2; reg v2;
  always @(posedge clk) begin v2 <= v1 & ~rst; if (v1) begin sum2 <= tree; e2 <= emax; end end
  // stage 3: accumulate, value = accm * 2^(acce - 254 - 22 - 12); no left normalisation inside the loop
  reg signed [47:0] accm; reg [9:0] acce;
  wire [9:0] e2x = {1'b0, e2};
  wire [9:0] er = (acce > e2x) ? acce : e2x;
  wire [9:0] da = er - acce, dt = er - e2x;
  wire signed [47:0] t = $signed({{7{sum2[28]}}, sum2, 12'd0});
  wire signed [47:0] ta = (dt >= 10'd48) ? 48'sd0 : (t >>> dt[5:0]);
  wire signed [47:0] aa = (da >= 10'd48) ? 48'sd0 : (accm >>> da[5:0]);
  wire signed [48:0] r = $signed({ta[47], ta}) + $signed({aa[47], aa});
  wire ovf = (r[48] != r[47]) | (r[47] != r[46]);
  always @(posedge clk) begin
    if (rst | mac_clr) begin accm <= 48'sd0; acce <= 10'd0; end
    else if (v2) begin accm <= ovf ? r[48:1] : r[47:0]; acce <= er + {9'd0, ovf}; end
  end
  // read-out: sign-magnitude, normalise, round to BF16
  wire sgn = accm[47];
  wire [47:0] mag = sgn ? (~accm + 48'd1) : accm;
  reg [5:0] lz; integer i;
  always @* begin lz = 6'd48; for (i = 0; i < 48; i = i + 1) if (mag[i]) lz = 6'd47 - i[5:0]; end
  wire [47:0] nm = mag << lz;
  // mag * 2^(acce - 288): leading one at bit 47 - lz -> biased exponent = acce - 288 + 47 - lz + 127
  wire signed [10:0] eo = $signed({1'b0, acce}) - 11'sd114 - $signed({5'd0, lz});
  bf16_pack p(.s(sgn), .e(eo), .m(nm[47:40]), .r(nm[39]), .z(mag == 48'd0), .y(mac_rd));
endmodule

// One datapath: 16 multipliers shared by MAC and element-wise multiply, 16 adders for element-wise add,
// a three-stage MAC (multiply | align + sum tree | accumulate), a read-out normaliser.
(* keep_hierarchy *)
module bf16_dp(
  input wire clk, input wire rst,
  input wire [255:0] a, input wire [255:0] b,
  input wire mac_v, input wire mac_clr,
  output wire [255:0] ewmul, output wire [255:0] ewadd, output wire [15:0] mac_rd);
  wire [15:0] ps; wire [143:0] pe; wire [255:0] pm;
  genvar g;
  generate for (g = 0; g < 16; g = g + 1) begin : L
    bf16_mulcore mc(.a(a[16*g +: 16]), .b(b[16*g +: 16]), .s(ps[g]), .e(pe[9*g +: 9]), .m(pm[16*g +: 16]));
    bf16_ewmul em(.s(ps[g]), .e(pe[9*g +: 9]), .m(pm[16*g +: 16]), .y(ewmul[16*g +: 16]));
    bf16_add ad(.a(a[16*g +: 16]), .b(b[16*g +: 16]), .y(ewadd[16*g +: 16]));
  end endgenerate
  reg [15:0] s1; reg [143:0] e1; reg [255:0] m1; reg v1;
  always @(posedge clk) begin v1 <= mac_v & ~rst; if (mac_v) begin s1 <= ps; e1 <= pe; m1 <= pm; end end
  bf16_acc acc(.clk(clk), .rst(rst), .v1(v1), .s1(s1), .e1(e1), .m1(m1), .mac_clr(mac_clr), .mac_rd(mac_rd));
endmodule
