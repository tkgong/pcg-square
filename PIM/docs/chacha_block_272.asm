; ============================================================================
; 一个 GGM/DPF ChaCha8 block(batch = 8 个 node 并行)完整指令清单
; 记号: n[i]      = batch 内第 i 个 node(住在 lane i,i=0..7)
;       n[i].w[k] = 该 node 128b seed 的第 k 个 32b 字(k=0..3)
;       s0..s15   = ChaCha 状态字;寄存器 Vj = {n[0].sj, n[1].sj, ..., n[7].sj}
;       每条 SIMD 指令同时作用全部 8 个 node 的同一个状态字
;       L[i]/R[i] = n[i] 的左/右孩子(各 128b = 4 字)
; 拍数: 计算 272 = 256 轮内 + 4 ff + 4 VCXOR + 4 VXOR + 4 PACKLR32
;       (12 条 VBCAST 融合进装载通路不计;显式则 272->284)
;       访存 = ACT×2(controller 发) + VLD×8 + VST×8(tCCD 节奏,另记账)
; ============================================================================

; ---- 阶段 0: DRAM 命令(controller 自动发,非 PU 指令) ----------------------
ACT    bankA, row_r          ; 打开父行; 行内: col c+k = {n[0].w[k] ... n[7].w[k]}
ACT    bankB, row_r2         ; 打开孩子行(ping-pong 对面 bank,写目标)
;                              (ACT 后等 tRCD=31 拍列才可读)

; ---- 阶段 1: 状态装载(4 VLD + 12 VBCAST) ----------------------------------
VBCAST V0,  0x61707865       ; s0  <- const "expa"      (n[0-7] 同值广播)
VBCAST V1,  0x3320646e       ; s1  <- const "nd 3"
VBCAST V2,  0x79622d32       ; s2  <- const "2-by"
VBCAST V3,  0x6b206574       ; s3  <- const "te k"
VLD    V4, (A,r,c+0)         ; s4  <- {n[0].w[0], n[1].w[0], ..., n[7].w[0]}
VLD    V5, (A,r,c+1)         ; s5  <- {n[0].w[1], n[1].w[1], ..., n[7].w[1]}
VLD    V6, (A,r,c+2)         ; s6  <- {n[0].w[2], n[1].w[2], ..., n[7].w[2]}
VLD    V7, (A,r,c+3)         ; s7  <- {n[0].w[3], n[1].w[3], ..., n[7].w[3]}
VBCAST V8,  0                ; s8  <- 0 (key 高位零填充)
VBCAST V9,  0                ; s9  <- 0
VBCAST V10, 0                ; s10 <- 0
VBCAST V11, 0                ; s11 <- 0
VBCAST V12, 0                ; s12 <- block counter = 0
VBCAST V13, 0                ; s13 <- nonce = 0
VBCAST V14, 0                ; s14 <- nonce = 0
VBCAST V15, 0                ; s15 <- nonce = 0
; SRF(常驻,本批不写): lane0-3 = 上面 4 常量; lane4-7 = 本层 CW.w[0..3]

; ---- 阶段 2: 轮函数 256 拍(4 个双轮 x 64 条) -------------------------------
VADD32  V0,V0,V4      ; 拍  1  [DR1 列QRA] n[0-7]: s0 += s4
VXROL16 V12,V12,V0    ; 拍  2  [DR1 列QRA] n[0-7]: s12 = rotl(s12^s0,16)
VADD32  V8,V8,V12     ; 拍  3  [DR1 列QRA] n[0-7]: s8 += s12
VXROL12 V4,V4,V8      ; 拍  4  [DR1 列QRA] n[0-7]: s4 = rotl(s4^s8,12)
VADD32  V0,V0,V4      ; 拍  5  [DR1 列QRA] n[0-7]: s0 += s4
VXROL8  V12,V12,V0    ; 拍  6  [DR1 列QRA] n[0-7]: s12 = rotl(s12^s0,8)
VADD32  V8,V8,V12     ; 拍  7  [DR1 列QRA] n[0-7]: s8 += s12
VXROL7  V4,V4,V8      ; 拍  8  [DR1 列QRA] n[0-7]: s4 = rotl(s4^s8,7)
VADD32  V1,V1,V5      ; 拍  9  [DR1 列QRB] n[0-7]: s1 += s5
VXROL16 V13,V13,V1    ; 拍 10  [DR1 列QRB] n[0-7]: s13 = rotl(s13^s1,16)
VADD32  V9,V9,V13     ; 拍 11  [DR1 列QRB] n[0-7]: s9 += s13
VXROL12 V5,V5,V9      ; 拍 12  [DR1 列QRB] n[0-7]: s5 = rotl(s5^s9,12)
VADD32  V1,V1,V5      ; 拍 13  [DR1 列QRB] n[0-7]: s1 += s5
VXROL8  V13,V13,V1    ; 拍 14  [DR1 列QRB] n[0-7]: s13 = rotl(s13^s1,8)
VADD32  V9,V9,V13     ; 拍 15  [DR1 列QRB] n[0-7]: s9 += s13
VXROL7  V5,V5,V9      ; 拍 16  [DR1 列QRB] n[0-7]: s5 = rotl(s5^s9,7)
VADD32  V2,V2,V6      ; 拍 17  [DR1 列QRC] n[0-7]: s2 += s6
VXROL16 V14,V14,V2    ; 拍 18  [DR1 列QRC] n[0-7]: s14 = rotl(s14^s2,16)
VADD32  V10,V10,V14   ; 拍 19  [DR1 列QRC] n[0-7]: s10 += s14
VXROL12 V6,V6,V10     ; 拍 20  [DR1 列QRC] n[0-7]: s6 = rotl(s6^s10,12)
VADD32  V2,V2,V6      ; 拍 21  [DR1 列QRC] n[0-7]: s2 += s6
VXROL8  V14,V14,V2    ; 拍 22  [DR1 列QRC] n[0-7]: s14 = rotl(s14^s2,8)
VADD32  V10,V10,V14   ; 拍 23  [DR1 列QRC] n[0-7]: s10 += s14
VXROL7  V6,V6,V10     ; 拍 24  [DR1 列QRC] n[0-7]: s6 = rotl(s6^s10,7)
VADD32  V3,V3,V7      ; 拍 25  [DR1 列QRD] n[0-7]: s3 += s7
VXROL16 V15,V15,V3    ; 拍 26  [DR1 列QRD] n[0-7]: s15 = rotl(s15^s3,16)
VADD32  V11,V11,V15   ; 拍 27  [DR1 列QRD] n[0-7]: s11 += s15
VXROL12 V7,V7,V11     ; 拍 28  [DR1 列QRD] n[0-7]: s7 = rotl(s7^s11,12)
VADD32  V3,V3,V7      ; 拍 29  [DR1 列QRD] n[0-7]: s3 += s7
VXROL8  V15,V15,V3    ; 拍 30  [DR1 列QRD] n[0-7]: s15 = rotl(s15^s3,8)
VADD32  V11,V11,V15   ; 拍 31  [DR1 列QRD] n[0-7]: s11 += s15
VXROL7  V7,V7,V11     ; 拍 32  [DR1 列QRD] n[0-7]: s7 = rotl(s7^s11,7)
VADD32  V0,V0,V5      ; 拍 33  [DR1 对角QRA] n[0-7]: s0 += s5
VXROL16 V15,V15,V0    ; 拍 34  [DR1 对角QRA] n[0-7]: s15 = rotl(s15^s0,16)
VADD32  V10,V10,V15   ; 拍 35  [DR1 对角QRA] n[0-7]: s10 += s15
VXROL12 V5,V5,V10     ; 拍 36  [DR1 对角QRA] n[0-7]: s5 = rotl(s5^s10,12)
VADD32  V0,V0,V5      ; 拍 37  [DR1 对角QRA] n[0-7]: s0 += s5
VXROL8  V15,V15,V0    ; 拍 38  [DR1 对角QRA] n[0-7]: s15 = rotl(s15^s0,8)
VADD32  V10,V10,V15   ; 拍 39  [DR1 对角QRA] n[0-7]: s10 += s15
VXROL7  V5,V5,V10     ; 拍 40  [DR1 对角QRA] n[0-7]: s5 = rotl(s5^s10,7)
VADD32  V1,V1,V6      ; 拍 41  [DR1 对角QRB] n[0-7]: s1 += s6
VXROL16 V12,V12,V1    ; 拍 42  [DR1 对角QRB] n[0-7]: s12 = rotl(s12^s1,16)
VADD32  V11,V11,V12   ; 拍 43  [DR1 对角QRB] n[0-7]: s11 += s12
VXROL12 V6,V6,V11     ; 拍 44  [DR1 对角QRB] n[0-7]: s6 = rotl(s6^s11,12)
VADD32  V1,V1,V6      ; 拍 45  [DR1 对角QRB] n[0-7]: s1 += s6
VXROL8  V12,V12,V1    ; 拍 46  [DR1 对角QRB] n[0-7]: s12 = rotl(s12^s1,8)
VADD32  V11,V11,V12   ; 拍 47  [DR1 对角QRB] n[0-7]: s11 += s12
VXROL7  V6,V6,V11     ; 拍 48  [DR1 对角QRB] n[0-7]: s6 = rotl(s6^s11,7)
VADD32  V2,V2,V7      ; 拍 49  [DR1 对角QRC] n[0-7]: s2 += s7
VXROL16 V13,V13,V2    ; 拍 50  [DR1 对角QRC] n[0-7]: s13 = rotl(s13^s2,16)
VADD32  V8,V8,V13     ; 拍 51  [DR1 对角QRC] n[0-7]: s8 += s13
VXROL12 V7,V7,V8      ; 拍 52  [DR1 对角QRC] n[0-7]: s7 = rotl(s7^s8,12)
VADD32  V2,V2,V7      ; 拍 53  [DR1 对角QRC] n[0-7]: s2 += s7
VXROL8  V13,V13,V2    ; 拍 54  [DR1 对角QRC] n[0-7]: s13 = rotl(s13^s2,8)
VADD32  V8,V8,V13     ; 拍 55  [DR1 对角QRC] n[0-7]: s8 += s13
VXROL7  V7,V7,V8      ; 拍 56  [DR1 对角QRC] n[0-7]: s7 = rotl(s7^s8,7)
VADD32  V3,V3,V4      ; 拍 57  [DR1 对角QRD] n[0-7]: s3 += s4
VXROL16 V14,V14,V3    ; 拍 58  [DR1 对角QRD] n[0-7]: s14 = rotl(s14^s3,16)
VADD32  V9,V9,V14     ; 拍 59  [DR1 对角QRD] n[0-7]: s9 += s14
VXROL12 V4,V4,V9      ; 拍 60  [DR1 对角QRD] n[0-7]: s4 = rotl(s4^s9,12)
VADD32  V3,V3,V4      ; 拍 61  [DR1 对角QRD] n[0-7]: s3 += s4
VXROL8  V14,V14,V3    ; 拍 62  [DR1 对角QRD] n[0-7]: s14 = rotl(s14^s3,8)
VADD32  V9,V9,V14     ; 拍 63  [DR1 对角QRD] n[0-7]: s9 += s14
VXROL7  V4,V4,V9      ; 拍 64  [DR1 对角QRD] n[0-7]: s4 = rotl(s4^s9,7)
VADD32  V0,V0,V4      ; 拍 65  [DR2 列QRA] n[0-7]: s0 += s4
VXROL16 V12,V12,V0    ; 拍 66  [DR2 列QRA] n[0-7]: s12 = rotl(s12^s0,16)
VADD32  V8,V8,V12     ; 拍 67  [DR2 列QRA] n[0-7]: s8 += s12
VXROL12 V4,V4,V8      ; 拍 68  [DR2 列QRA] n[0-7]: s4 = rotl(s4^s8,12)
VADD32  V0,V0,V4      ; 拍 69  [DR2 列QRA] n[0-7]: s0 += s4
VXROL8  V12,V12,V0    ; 拍 70  [DR2 列QRA] n[0-7]: s12 = rotl(s12^s0,8)
VADD32  V8,V8,V12     ; 拍 71  [DR2 列QRA] n[0-7]: s8 += s12
VXROL7  V4,V4,V8      ; 拍 72  [DR2 列QRA] n[0-7]: s4 = rotl(s4^s8,7)
VADD32  V1,V1,V5      ; 拍 73  [DR2 列QRB] n[0-7]: s1 += s5
VXROL16 V13,V13,V1    ; 拍 74  [DR2 列QRB] n[0-7]: s13 = rotl(s13^s1,16)
VADD32  V9,V9,V13     ; 拍 75  [DR2 列QRB] n[0-7]: s9 += s13
VXROL12 V5,V5,V9      ; 拍 76  [DR2 列QRB] n[0-7]: s5 = rotl(s5^s9,12)
VADD32  V1,V1,V5      ; 拍 77  [DR2 列QRB] n[0-7]: s1 += s5
VXROL8  V13,V13,V1    ; 拍 78  [DR2 列QRB] n[0-7]: s13 = rotl(s13^s1,8)
VADD32  V9,V9,V13     ; 拍 79  [DR2 列QRB] n[0-7]: s9 += s13
VXROL7  V5,V5,V9      ; 拍 80  [DR2 列QRB] n[0-7]: s5 = rotl(s5^s9,7)
VADD32  V2,V2,V6      ; 拍 81  [DR2 列QRC] n[0-7]: s2 += s6
VXROL16 V14,V14,V2    ; 拍 82  [DR2 列QRC] n[0-7]: s14 = rotl(s14^s2,16)
VADD32  V10,V10,V14   ; 拍 83  [DR2 列QRC] n[0-7]: s10 += s14
VXROL12 V6,V6,V10     ; 拍 84  [DR2 列QRC] n[0-7]: s6 = rotl(s6^s10,12)
VADD32  V2,V2,V6      ; 拍 85  [DR2 列QRC] n[0-7]: s2 += s6
VXROL8  V14,V14,V2    ; 拍 86  [DR2 列QRC] n[0-7]: s14 = rotl(s14^s2,8)
VADD32  V10,V10,V14   ; 拍 87  [DR2 列QRC] n[0-7]: s10 += s14
VXROL7  V6,V6,V10     ; 拍 88  [DR2 列QRC] n[0-7]: s6 = rotl(s6^s10,7)
VADD32  V3,V3,V7      ; 拍 89  [DR2 列QRD] n[0-7]: s3 += s7
VXROL16 V15,V15,V3    ; 拍 90  [DR2 列QRD] n[0-7]: s15 = rotl(s15^s3,16)
VADD32  V11,V11,V15   ; 拍 91  [DR2 列QRD] n[0-7]: s11 += s15
VXROL12 V7,V7,V11     ; 拍 92  [DR2 列QRD] n[0-7]: s7 = rotl(s7^s11,12)
VADD32  V3,V3,V7      ; 拍 93  [DR2 列QRD] n[0-7]: s3 += s7
VXROL8  V15,V15,V3    ; 拍 94  [DR2 列QRD] n[0-7]: s15 = rotl(s15^s3,8)
VADD32  V11,V11,V15   ; 拍 95  [DR2 列QRD] n[0-7]: s11 += s15
VXROL7  V7,V7,V11     ; 拍 96  [DR2 列QRD] n[0-7]: s7 = rotl(s7^s11,7)
VADD32  V0,V0,V5      ; 拍 97  [DR2 对角QRA] n[0-7]: s0 += s5
VXROL16 V15,V15,V0    ; 拍 98  [DR2 对角QRA] n[0-7]: s15 = rotl(s15^s0,16)
VADD32  V10,V10,V15   ; 拍 99  [DR2 对角QRA] n[0-7]: s10 += s15
VXROL12 V5,V5,V10     ; 拍100  [DR2 对角QRA] n[0-7]: s5 = rotl(s5^s10,12)
VADD32  V0,V0,V5      ; 拍101  [DR2 对角QRA] n[0-7]: s0 += s5
VXROL8  V15,V15,V0    ; 拍102  [DR2 对角QRA] n[0-7]: s15 = rotl(s15^s0,8)
VADD32  V10,V10,V15   ; 拍103  [DR2 对角QRA] n[0-7]: s10 += s15
VXROL7  V5,V5,V10     ; 拍104  [DR2 对角QRA] n[0-7]: s5 = rotl(s5^s10,7)
VADD32  V1,V1,V6      ; 拍105  [DR2 对角QRB] n[0-7]: s1 += s6
VXROL16 V12,V12,V1    ; 拍106  [DR2 对角QRB] n[0-7]: s12 = rotl(s12^s1,16)
VADD32  V11,V11,V12   ; 拍107  [DR2 对角QRB] n[0-7]: s11 += s12
VXROL12 V6,V6,V11     ; 拍108  [DR2 对角QRB] n[0-7]: s6 = rotl(s6^s11,12)
VADD32  V1,V1,V6      ; 拍109  [DR2 对角QRB] n[0-7]: s1 += s6
VXROL8  V12,V12,V1    ; 拍110  [DR2 对角QRB] n[0-7]: s12 = rotl(s12^s1,8)
VADD32  V11,V11,V12   ; 拍111  [DR2 对角QRB] n[0-7]: s11 += s12
VXROL7  V6,V6,V11     ; 拍112  [DR2 对角QRB] n[0-7]: s6 = rotl(s6^s11,7)
VADD32  V2,V2,V7      ; 拍113  [DR2 对角QRC] n[0-7]: s2 += s7
VXROL16 V13,V13,V2    ; 拍114  [DR2 对角QRC] n[0-7]: s13 = rotl(s13^s2,16)
VADD32  V8,V8,V13     ; 拍115  [DR2 对角QRC] n[0-7]: s8 += s13
VXROL12 V7,V7,V8      ; 拍116  [DR2 对角QRC] n[0-7]: s7 = rotl(s7^s8,12)
VADD32  V2,V2,V7      ; 拍117  [DR2 对角QRC] n[0-7]: s2 += s7
VXROL8  V13,V13,V2    ; 拍118  [DR2 对角QRC] n[0-7]: s13 = rotl(s13^s2,8)
VADD32  V8,V8,V13     ; 拍119  [DR2 对角QRC] n[0-7]: s8 += s13
VXROL7  V7,V7,V8      ; 拍120  [DR2 对角QRC] n[0-7]: s7 = rotl(s7^s8,7)
VADD32  V3,V3,V4      ; 拍121  [DR2 对角QRD] n[0-7]: s3 += s4
VXROL16 V14,V14,V3    ; 拍122  [DR2 对角QRD] n[0-7]: s14 = rotl(s14^s3,16)
VADD32  V9,V9,V14     ; 拍123  [DR2 对角QRD] n[0-7]: s9 += s14
VXROL12 V4,V4,V9      ; 拍124  [DR2 对角QRD] n[0-7]: s4 = rotl(s4^s9,12)
VADD32  V3,V3,V4      ; 拍125  [DR2 对角QRD] n[0-7]: s3 += s4
VXROL8  V14,V14,V3    ; 拍126  [DR2 对角QRD] n[0-7]: s14 = rotl(s14^s3,8)
VADD32  V9,V9,V14     ; 拍127  [DR2 对角QRD] n[0-7]: s9 += s14
VXROL7  V4,V4,V9      ; 拍128  [DR2 对角QRD] n[0-7]: s4 = rotl(s4^s9,7)
VADD32  V0,V0,V4      ; 拍129  [DR3 列QRA] n[0-7]: s0 += s4
VXROL16 V12,V12,V0    ; 拍130  [DR3 列QRA] n[0-7]: s12 = rotl(s12^s0,16)
VADD32  V8,V8,V12     ; 拍131  [DR3 列QRA] n[0-7]: s8 += s12
VXROL12 V4,V4,V8      ; 拍132  [DR3 列QRA] n[0-7]: s4 = rotl(s4^s8,12)
VADD32  V0,V0,V4      ; 拍133  [DR3 列QRA] n[0-7]: s0 += s4
VXROL8  V12,V12,V0    ; 拍134  [DR3 列QRA] n[0-7]: s12 = rotl(s12^s0,8)
VADD32  V8,V8,V12     ; 拍135  [DR3 列QRA] n[0-7]: s8 += s12
VXROL7  V4,V4,V8      ; 拍136  [DR3 列QRA] n[0-7]: s4 = rotl(s4^s8,7)
VADD32  V1,V1,V5      ; 拍137  [DR3 列QRB] n[0-7]: s1 += s5
VXROL16 V13,V13,V1    ; 拍138  [DR3 列QRB] n[0-7]: s13 = rotl(s13^s1,16)
VADD32  V9,V9,V13     ; 拍139  [DR3 列QRB] n[0-7]: s9 += s13
VXROL12 V5,V5,V9      ; 拍140  [DR3 列QRB] n[0-7]: s5 = rotl(s5^s9,12)
VADD32  V1,V1,V5      ; 拍141  [DR3 列QRB] n[0-7]: s1 += s5
VXROL8  V13,V13,V1    ; 拍142  [DR3 列QRB] n[0-7]: s13 = rotl(s13^s1,8)
VADD32  V9,V9,V13     ; 拍143  [DR3 列QRB] n[0-7]: s9 += s13
VXROL7  V5,V5,V9      ; 拍144  [DR3 列QRB] n[0-7]: s5 = rotl(s5^s9,7)
VADD32  V2,V2,V6      ; 拍145  [DR3 列QRC] n[0-7]: s2 += s6
VXROL16 V14,V14,V2    ; 拍146  [DR3 列QRC] n[0-7]: s14 = rotl(s14^s2,16)
VADD32  V10,V10,V14   ; 拍147  [DR3 列QRC] n[0-7]: s10 += s14
VXROL12 V6,V6,V10     ; 拍148  [DR3 列QRC] n[0-7]: s6 = rotl(s6^s10,12)
VADD32  V2,V2,V6      ; 拍149  [DR3 列QRC] n[0-7]: s2 += s6
VXROL8  V14,V14,V2    ; 拍150  [DR3 列QRC] n[0-7]: s14 = rotl(s14^s2,8)
VADD32  V10,V10,V14   ; 拍151  [DR3 列QRC] n[0-7]: s10 += s14
VXROL7  V6,V6,V10     ; 拍152  [DR3 列QRC] n[0-7]: s6 = rotl(s6^s10,7)
VADD32  V3,V3,V7      ; 拍153  [DR3 列QRD] n[0-7]: s3 += s7
VXROL16 V15,V15,V3    ; 拍154  [DR3 列QRD] n[0-7]: s15 = rotl(s15^s3,16)
VADD32  V11,V11,V15   ; 拍155  [DR3 列QRD] n[0-7]: s11 += s15
VXROL12 V7,V7,V11     ; 拍156  [DR3 列QRD] n[0-7]: s7 = rotl(s7^s11,12)
VADD32  V3,V3,V7      ; 拍157  [DR3 列QRD] n[0-7]: s3 += s7
VXROL8  V15,V15,V3    ; 拍158  [DR3 列QRD] n[0-7]: s15 = rotl(s15^s3,8)
VADD32  V11,V11,V15   ; 拍159  [DR3 列QRD] n[0-7]: s11 += s15
VXROL7  V7,V7,V11     ; 拍160  [DR3 列QRD] n[0-7]: s7 = rotl(s7^s11,7)
VADD32  V0,V0,V5      ; 拍161  [DR3 对角QRA] n[0-7]: s0 += s5
VXROL16 V15,V15,V0    ; 拍162  [DR3 对角QRA] n[0-7]: s15 = rotl(s15^s0,16)
VADD32  V10,V10,V15   ; 拍163  [DR3 对角QRA] n[0-7]: s10 += s15
VXROL12 V5,V5,V10     ; 拍164  [DR3 对角QRA] n[0-7]: s5 = rotl(s5^s10,12)
VADD32  V0,V0,V5      ; 拍165  [DR3 对角QRA] n[0-7]: s0 += s5
VXROL8  V15,V15,V0    ; 拍166  [DR3 对角QRA] n[0-7]: s15 = rotl(s15^s0,8)
VADD32  V10,V10,V15   ; 拍167  [DR3 对角QRA] n[0-7]: s10 += s15
VXROL7  V5,V5,V10     ; 拍168  [DR3 对角QRA] n[0-7]: s5 = rotl(s5^s10,7)
VADD32  V1,V1,V6      ; 拍169  [DR3 对角QRB] n[0-7]: s1 += s6
VXROL16 V12,V12,V1    ; 拍170  [DR3 对角QRB] n[0-7]: s12 = rotl(s12^s1,16)
VADD32  V11,V11,V12   ; 拍171  [DR3 对角QRB] n[0-7]: s11 += s12
VXROL12 V6,V6,V11     ; 拍172  [DR3 对角QRB] n[0-7]: s6 = rotl(s6^s11,12)
VADD32  V1,V1,V6      ; 拍173  [DR3 对角QRB] n[0-7]: s1 += s6
VXROL8  V12,V12,V1    ; 拍174  [DR3 对角QRB] n[0-7]: s12 = rotl(s12^s1,8)
VADD32  V11,V11,V12   ; 拍175  [DR3 对角QRB] n[0-7]: s11 += s12
VXROL7  V6,V6,V11     ; 拍176  [DR3 对角QRB] n[0-7]: s6 = rotl(s6^s11,7)
VADD32  V2,V2,V7      ; 拍177  [DR3 对角QRC] n[0-7]: s2 += s7
VXROL16 V13,V13,V2    ; 拍178  [DR3 对角QRC] n[0-7]: s13 = rotl(s13^s2,16)
VADD32  V8,V8,V13     ; 拍179  [DR3 对角QRC] n[0-7]: s8 += s13
VXROL12 V7,V7,V8      ; 拍180  [DR3 对角QRC] n[0-7]: s7 = rotl(s7^s8,12)
VADD32  V2,V2,V7      ; 拍181  [DR3 对角QRC] n[0-7]: s2 += s7
VXROL8  V13,V13,V2    ; 拍182  [DR3 对角QRC] n[0-7]: s13 = rotl(s13^s2,8)
VADD32  V8,V8,V13     ; 拍183  [DR3 对角QRC] n[0-7]: s8 += s13
VXROL7  V7,V7,V8      ; 拍184  [DR3 对角QRC] n[0-7]: s7 = rotl(s7^s8,7)
VADD32  V3,V3,V4      ; 拍185  [DR3 对角QRD] n[0-7]: s3 += s4
VXROL16 V14,V14,V3    ; 拍186  [DR3 对角QRD] n[0-7]: s14 = rotl(s14^s3,16)
VADD32  V9,V9,V14     ; 拍187  [DR3 对角QRD] n[0-7]: s9 += s14
VXROL12 V4,V4,V9      ; 拍188  [DR3 对角QRD] n[0-7]: s4 = rotl(s4^s9,12)
VADD32  V3,V3,V4      ; 拍189  [DR3 对角QRD] n[0-7]: s3 += s4
VXROL8  V14,V14,V3    ; 拍190  [DR3 对角QRD] n[0-7]: s14 = rotl(s14^s3,8)
VADD32  V9,V9,V14     ; 拍191  [DR3 对角QRD] n[0-7]: s9 += s14
VXROL7  V4,V4,V9      ; 拍192  [DR3 对角QRD] n[0-7]: s4 = rotl(s4^s9,7)
VADD32  V0,V0,V4      ; 拍193  [DR4 列QRA] n[0-7]: s0 += s4
VXROL16 V12,V12,V0    ; 拍194  [DR4 列QRA] n[0-7]: s12 = rotl(s12^s0,16)
VADD32  V8,V8,V12     ; 拍195  [DR4 列QRA] n[0-7]: s8 += s12
VXROL12 V4,V4,V8      ; 拍196  [DR4 列QRA] n[0-7]: s4 = rotl(s4^s8,12)
VADD32  V0,V0,V4      ; 拍197  [DR4 列QRA] n[0-7]: s0 += s4
VXROL8  V12,V12,V0    ; 拍198  [DR4 列QRA] n[0-7]: s12 = rotl(s12^s0,8)
VADD32  V8,V8,V12     ; 拍199  [DR4 列QRA] n[0-7]: s8 += s12
VXROL7  V4,V4,V8      ; 拍200  [DR4 列QRA] n[0-7]: s4 = rotl(s4^s8,7)
VADD32  V1,V1,V5      ; 拍201  [DR4 列QRB] n[0-7]: s1 += s5
VXROL16 V13,V13,V1    ; 拍202  [DR4 列QRB] n[0-7]: s13 = rotl(s13^s1,16)
VADD32  V9,V9,V13     ; 拍203  [DR4 列QRB] n[0-7]: s9 += s13
VXROL12 V5,V5,V9      ; 拍204  [DR4 列QRB] n[0-7]: s5 = rotl(s5^s9,12)
VADD32  V1,V1,V5      ; 拍205  [DR4 列QRB] n[0-7]: s1 += s5
VXROL8  V13,V13,V1    ; 拍206  [DR4 列QRB] n[0-7]: s13 = rotl(s13^s1,8)
VADD32  V9,V9,V13     ; 拍207  [DR4 列QRB] n[0-7]: s9 += s13
VXROL7  V5,V5,V9      ; 拍208  [DR4 列QRB] n[0-7]: s5 = rotl(s5^s9,7)
VADD32  V2,V2,V6      ; 拍209  [DR4 列QRC] n[0-7]: s2 += s6
VXROL16 V14,V14,V2    ; 拍210  [DR4 列QRC] n[0-7]: s14 = rotl(s14^s2,16)
VADD32  V10,V10,V14   ; 拍211  [DR4 列QRC] n[0-7]: s10 += s14
VXROL12 V6,V6,V10     ; 拍212  [DR4 列QRC] n[0-7]: s6 = rotl(s6^s10,12)
VADD32  V2,V2,V6      ; 拍213  [DR4 列QRC] n[0-7]: s2 += s6
VXROL8  V14,V14,V2    ; 拍214  [DR4 列QRC] n[0-7]: s14 = rotl(s14^s2,8)
VADD32  V10,V10,V14   ; 拍215  [DR4 列QRC] n[0-7]: s10 += s14
VXROL7  V6,V6,V10     ; 拍216  [DR4 列QRC] n[0-7]: s6 = rotl(s6^s10,7)
VADD32  V3,V3,V7      ; 拍217  [DR4 列QRD] n[0-7]: s3 += s7
VXROL16 V15,V15,V3    ; 拍218  [DR4 列QRD] n[0-7]: s15 = rotl(s15^s3,16)
VADD32  V11,V11,V15   ; 拍219  [DR4 列QRD] n[0-7]: s11 += s15
VXROL12 V7,V7,V11     ; 拍220  [DR4 列QRD] n[0-7]: s7 = rotl(s7^s11,12)
VADD32  V3,V3,V7      ; 拍221  [DR4 列QRD] n[0-7]: s3 += s7
VXROL8  V15,V15,V3    ; 拍222  [DR4 列QRD] n[0-7]: s15 = rotl(s15^s3,8)
VADD32  V11,V11,V15   ; 拍223  [DR4 列QRD] n[0-7]: s11 += s15
VXROL7  V7,V7,V11     ; 拍224  [DR4 列QRD] n[0-7]: s7 = rotl(s7^s11,7)
VADD32  V0,V0,V5      ; 拍225  [DR4 对角QRA] n[0-7]: s0 += s5
VXROL16 V15,V15,V0    ; 拍226  [DR4 对角QRA] n[0-7]: s15 = rotl(s15^s0,16)
VADD32  V10,V10,V15   ; 拍227  [DR4 对角QRA] n[0-7]: s10 += s15
VXROL12 V5,V5,V10     ; 拍228  [DR4 对角QRA] n[0-7]: s5 = rotl(s5^s10,12)
VADD32  V0,V0,V5      ; 拍229  [DR4 对角QRA] n[0-7]: s0 += s5
VXROL8  V15,V15,V0    ; 拍230  [DR4 对角QRA] n[0-7]: s15 = rotl(s15^s0,8)
VADD32  V10,V10,V15   ; 拍231  [DR4 对角QRA] n[0-7]: s10 += s15
VXROL7  V5,V5,V10     ; 拍232  [DR4 对角QRA] n[0-7]: s5 = rotl(s5^s10,7)
VADD32  V1,V1,V6      ; 拍233  [DR4 对角QRB] n[0-7]: s1 += s6
VXROL16 V12,V12,V1    ; 拍234  [DR4 对角QRB] n[0-7]: s12 = rotl(s12^s1,16)
VADD32  V11,V11,V12   ; 拍235  [DR4 对角QRB] n[0-7]: s11 += s12
VXROL12 V6,V6,V11     ; 拍236  [DR4 对角QRB] n[0-7]: s6 = rotl(s6^s11,12)
VADD32  V1,V1,V6      ; 拍237  [DR4 对角QRB] n[0-7]: s1 += s6
VXROL8  V12,V12,V1    ; 拍238  [DR4 对角QRB] n[0-7]: s12 = rotl(s12^s1,8)
VADD32  V11,V11,V12   ; 拍239  [DR4 对角QRB] n[0-7]: s11 += s12
VXROL7  V6,V6,V11     ; 拍240  [DR4 对角QRB] n[0-7]: s6 = rotl(s6^s11,7)
VADD32  V2,V2,V7      ; 拍241  [DR4 对角QRC] n[0-7]: s2 += s7
VXROL16 V13,V13,V2    ; 拍242  [DR4 对角QRC] n[0-7]: s13 = rotl(s13^s2,16)
VADD32  V8,V8,V13     ; 拍243  [DR4 对角QRC] n[0-7]: s8 += s13
VXROL12 V7,V7,V8      ; 拍244  [DR4 对角QRC] n[0-7]: s7 = rotl(s7^s8,12)
VADD32  V2,V2,V7      ; 拍245  [DR4 对角QRC] n[0-7]: s2 += s7
VXROL8  V13,V13,V2    ; 拍246  [DR4 对角QRC] n[0-7]: s13 = rotl(s13^s2,8)
VADD32  V8,V8,V13     ; 拍247  [DR4 对角QRC] n[0-7]: s8 += s13
VXROL7  V7,V7,V8      ; 拍248  [DR4 对角QRC] n[0-7]: s7 = rotl(s7^s8,7)
VADD32  V3,V3,V4      ; 拍249  [DR4 对角QRD] n[0-7]: s3 += s4
VXROL16 V14,V14,V3    ; 拍250  [DR4 对角QRD] n[0-7]: s14 = rotl(s14^s3,16)
VADD32  V9,V9,V14     ; 拍251  [DR4 对角QRD] n[0-7]: s9 += s14
VXROL12 V4,V4,V9      ; 拍252  [DR4 对角QRD] n[0-7]: s4 = rotl(s4^s9,12)
VADD32  V3,V3,V4      ; 拍253  [DR4 对角QRD] n[0-7]: s3 += s4
VXROL8  V14,V14,V3    ; 拍254  [DR4 对角QRD] n[0-7]: s14 = rotl(s14^s3,8)
VADD32  V9,V9,V14     ; 拍255  [DR4 对角QRD] n[0-7]: s9 += s14
VXROL7  V4,V4,V9      ; 拍256  [DR4 对角QRD] n[0-7]: s4 = rotl(s4^s9,7)

; ---- 阶段 3: feed-forward(4 拍,只加被消费的 4 个字) -----------------------
VADD32 V0,V0,SRF[0]          ; 拍257  n[0-7]: h.w[0] = s0 + const0   (keystream 字0)
VADD32 V1,V1,SRF[1]          ; 拍258  n[0-7]: h.w[1] = s1 + const1
VADD32 V2,V2,SRF[2]          ; 拍259  n[0-7]: h.w[2] = s2 + const2
VADD32 V3,V3,SRF[3]          ; 拍260  n[0-7]: h.w[3] = s3 + const3
;      此刻 V4-V15 = 搅烂的 s4-s15,再也不用 -> 死寄存器

; ---- 阶段 4: 重读 parent(4 VLD,行仍开着,免 ACT;落进死寄存器) ------------
VLD    V4, (A,r,c+0)         ; s4 <- {n[0].w[0] ... n[7].w[0]}  原始 seed 回来
VLD    V5, (A,r,c+1)         ; s5 <- {n[0].w[1] ... n[7].w[1]}
VLD    V6, (A,r,c+2)         ; s6 <- {n[0].w[2] ... n[7].w[2]}
VLD    V7, (A,r,c+3)         ; s7 <- {n[0].w[3] ... n[7].w[3]}

; ---- 阶段 5: 出左右孩子(4 VCXOR + 4 VXOR) ---------------------------------
; t_i = lsb(n[i].w[0]) = node i 的 control bit(逐 lane 取自 V4)
VCXOR  V0,SRF[4],V4          ; 拍261  n[0-7]: L[i].w[0] = h.w[0] ^ t_i*CW.w[0]
VCXOR  V1,SRF[5],V4          ; 拍262  n[0-7]: L[i].w[1] = h.w[1] ^ t_i*CW.w[1]
VCXOR  V2,SRF[6],V4          ; 拍263  n[0-7]: L[i].w[2] = h.w[2] ^ t_i*CW.w[2]
VCXOR  V3,SRF[7],V4          ; 拍264  n[0-7]: L[i].w[3] = h.w[3] ^ t_i*CW.w[3]
VXOR   V4,V4,V0              ; 拍265  n[0-7]: R[i].w[0] = n[i].w[0] ^ L[i].w[0]
VXOR   V5,V5,V1              ; 拍266  n[0-7]: R[i].w[1] = n[i].w[1] ^ L[i].w[1]
VXOR   V6,V6,V2              ; 拍267  n[0-7]: R[i].w[2] = n[i].w[2] ^ L[i].w[2]
VXOR   V7,V7,V3              ; 拍268  n[0-7]: R[i].w[3] = n[i].w[3] ^ L[i].w[3]
; (w=1 时两孩子 CW 相同 -> R = parent ^ L,第二次 CW 已折入)

; ---- 阶段 6: 跨 lane 交织(4 PACKLR32),孩子按树序排列 ----------------------
; 新一层 node 编号 m[2i]=L[i], m[2i+1]=R[i]
PACKLR32 V0,V4               ; 拍269  V0={L0.w0,R0.w0,L1.w0,R1.w0,L2.w0,R2.w0,L3.w0,R3.w0}
                             ;        V4={L4.w0,R4.w0,L5.w0,R5.w0,L6.w0,R6.w0,L7.w0,R7.w0}
PACKLR32 V1,V5               ; 拍270  同上,字1: V1=m[0-7].w[1], V5=m[8-15].w[1]
PACKLR32 V2,V6               ; 拍271  字2: V2=m[0-7].w[2], V6=m[8-15].w[2]
PACKLR32 V3,V7               ; 拍272  字3: V3=m[0-7].w[3], V7=m[8-15].w[3]

; ---- 阶段 7: 写回(8 VST,访存账) -------------------------------------------
VST (B,r2,c2+0), V0          ; col <- {m[0].w[0] ... m[7].w[0]}   新组0 字0
VST (B,r2,c2+1), V1          ; col <- {m[0].w[1] ... m[7].w[1]}
VST (B,r2,c2+2), V2          ; col <- {m[0].w[2] ... m[7].w[2]}
VST (B,r2,c2+3), V3          ; col <- {m[0].w[3] ... m[7].w[3]}
VST (B,r2,c2+4), V4          ; col <- {m[8].w[0] ... m[15].w[0]}  新组1 字0
VST (B,r2,c2+5), V5          ; col <- {m[8].w[1] ... m[15].w[1]}
VST (B,r2,c2+6), V6          ; col <- {m[8].w[2] ... m[15].w[2]}
VST (B,r2,c2+7), V7          ; col <- {m[8].w[3] ... m[15].w[3]}
; ============================================================================
