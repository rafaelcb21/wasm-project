(module
  (memory (export "memory") 55)

  ;; ===== AUTO-GERADO: weights/bias RAW + MUL/Q6 + params + slots =====
  ;; MODEL: mobilenetv2_alpha035_quant.tflite
  ;; LayerParam size: 112 bytes | layers: 65

  ;; IMPORTANTE: Para operações multi-input (ADD, etc):
  ;;   - pad_t = input_ptr[0] (ponteiro para slot do primeiro input)
  ;;   - pad_b = input_ptr[1] (ponteiro para slot do segundo input)
  ;;   - pad_l = input_ptr[2] (ponteiro para slot do terceiro input)
  ;;   - pad_r = input_ptr[3] (ponteiro para slot do quarto input)

  ;; --- BASES (no overlap) ---
  (global $PARAMS_BASE i32 (i32.const 1760576))

  ;; --- layer list (ALL ops) — FULL DUMP ---
  ;; ================== L0 ==================
  ;; op_index          : 1
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 224 x 224
  ;; cin/cout          : 3 -> 16
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 2 x 2
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 1 0 1
  ;; out_h/out_w       : 112 x 112
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 1664096 / 1696256 / 1728416
  ;; zx/zw/zy           : -1 / 0 / -128
  ;;
  ;; ================== L1 ==================
  ;; op_index          : 2
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 112 x 112
  ;; cin/cout          : 16 -> 16
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 112 x 112
  ;; depth_mult        : 1
  ;; w_off/b_off       : 432 / 64
  ;; mul_off/q6_off     : 64 / 64
  ;; wptr/bias/mul/q6   : 2480 / 1664160 / 1696320 / 1728480
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L2 ==================
  ;; op_index          : 3
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 112 x 112
  ;; cin/cout          : 16 -> 8
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 112 x 112
  ;; w_off/b_off       : 576 / 128
  ;; mul_off/q6_off     : 128 / 128
  ;; wptr/bias/mul/q6   : 2624 / 1664224 / 1696384 / 0
  ;; zx/zw/zy           : -128 / 0 / -24
  ;;
  ;; ================== L3 ==================
  ;; op_index          : 4
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 112 x 112
  ;; cin/cout          : 8 -> 48
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 112 x 112
  ;; w_off/b_off       : 704 / 160
  ;; mul_off/q6_off     : 160 / 160
  ;; wptr/bias/mul/q6   : 2752 / 1664256 / 1696416 / 1728576
  ;; zx/zw/zy           : -24 / 0 / -128
  ;;
  ;; ================== L4 ==================
  ;; op_index          : 5
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 112 x 112
  ;; cin/cout          : 48 -> 48
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 2 x 2
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 1 0 1
  ;; out_h/out_w       : 56 x 56
  ;; depth_mult        : 1
  ;; w_off/b_off       : 1088 / 352
  ;; mul_off/q6_off     : 352 / 352
  ;; wptr/bias/mul/q6   : 3136 / 1664448 / 1696608 / 1728768
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L5 ==================
  ;; op_index          : 6
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 56 x 56
  ;; cin/cout          : 48 -> 8
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 56 x 56
  ;; w_off/b_off       : 1520 / 544
  ;; mul_off/q6_off     : 544 / 544
  ;; wptr/bias/mul/q6   : 3568 / 1664640 / 1696800 / 0
  ;; zx/zw/zy           : -128 / 0 / -3
  ;;
  ;; ================== L6 ==================
  ;; op_index          : 7
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 56 x 56
  ;; cin/cout          : 8 -> 48
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 56 x 56
  ;; w_off/b_off       : 1904 / 576
  ;; mul_off/q6_off     : 576 / 576
  ;; wptr/bias/mul/q6   : 3952 / 1664672 / 1696832 / 1728992
  ;; zx/zw/zy           : -3 / 0 / -128
  ;;
  ;; ================== L7 ==================
  ;; op_index          : 8
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 56 x 56
  ;; cin/cout          : 48 -> 48
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 56 x 56
  ;; depth_mult        : 1
  ;; w_off/b_off       : 2288 / 768
  ;; mul_off/q6_off     : 768 / 768
  ;; wptr/bias/mul/q6   : 4336 / 1664864 / 1697024 / 1729184
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L8 ==================
  ;; op_index          : 9
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 1
  ;; in_ptr/out_ptr    : 2972080 -> 2369968
  ;; in_h/in_w         : 56 x 56
  ;; cin/cout          : 48 -> 8
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 56 x 56
  ;; w_off/b_off       : 2720 / 960
  ;; mul_off/q6_off     : 960 / 960
  ;; wptr/bias/mul/q6   : 4768 / 1665056 / 1697216 / 0
  ;; zx/zw/zy           : -128 / 0 / -11
  ;;
  ;; ================== L9 ==================
  ;; op_index          : 10
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [0 e 1] -> 2
  ;; in_ptr/out_ptr    : 1767856 -> 2972080
  ;; input_ptrs        : [1767856, 2369968]
  ;; in_h/in_w         : 56 x 56
  ;; cin/cout          : 8 -> 8
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1767856 2369968 0 0 (input_ptrs)
  ;; out_h/out_w       : 56 x 56
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : -3 / 0 / -11
  ;;
  ;; ================== L10 ==================
  ;; op_index          : 11
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 56 x 56
  ;; cin/cout          : 8 -> 48
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 56 x 56
  ;; w_off/b_off       : 3104 / 992
  ;; mul_off/q6_off     : 992 / 992
  ;; wptr/bias/mul/q6   : 5152 / 1665088 / 1697248 / 1729408
  ;; zx/zw/zy           : -11 / 0 / -128
  ;;
  ;; ================== L11 ==================
  ;; op_index          : 12
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 56 x 56
  ;; cin/cout          : 48 -> 48
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 2 x 2
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 1 0 1
  ;; out_h/out_w       : 28 x 28
  ;; depth_mult        : 1
  ;; w_off/b_off       : 3488 / 1184
  ;; mul_off/q6_off     : 1184 / 1184
  ;; wptr/bias/mul/q6   : 5536 / 1665280 / 1697440 / 1729600
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L12 ==================
  ;; op_index          : 13
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 48 -> 16
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 3920 / 1376
  ;; mul_off/q6_off     : 1376 / 1376
  ;; wptr/bias/mul/q6   : 5968 / 1665472 / 1697632 / 0
  ;; zx/zw/zy           : -128 / 0 / -4
  ;;
  ;; ================== L13 ==================
  ;; op_index          : 14
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 16 -> 96
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 4688 / 1440
  ;; mul_off/q6_off     : 1440 / 1440
  ;; wptr/bias/mul/q6   : 6736 / 1665536 / 1697696 / 1729856
  ;; zx/zw/zy           : -4 / 0 / -128
  ;;
  ;; ================== L14 ==================
  ;; op_index          : 15
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 96 -> 96
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 28 x 28
  ;; depth_mult        : 1
  ;; w_off/b_off       : 6224 / 1824
  ;; mul_off/q6_off     : 1824 / 1824
  ;; wptr/bias/mul/q6   : 8272 / 1665920 / 1698080 / 1730240
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L15 ==================
  ;; op_index          : 16
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 1 -> 0
  ;; in_ptr/out_ptr    : 2369968 -> 1767856
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 96 -> 16
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 7088 / 2208
  ;; mul_off/q6_off     : 2208 / 2208
  ;; wptr/bias/mul/q6   : 9136 / 1666304 / 1698464 / 0
  ;; zx/zw/zy           : -128 / 0 / 16
  ;;
  ;; ================== L16 ==================
  ;; op_index          : 17
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [2 e 0] -> 1
  ;; in_ptr/out_ptr    : 2972080 -> 2369968
  ;; input_ptrs        : [2972080, 1767856]
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 16 -> 16
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 2972080 1767856 0 0 (input_ptrs)
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : -4 / 0 / 16
  ;;
  ;; ================== L17 ==================
  ;; op_index          : 18
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 16 -> 96
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 8624 / 2272
  ;; mul_off/q6_off     : 2272 / 2272
  ;; wptr/bias/mul/q6   : 10672 / 1666368 / 1698528 / 1730688
  ;; zx/zw/zy           : 16 / 0 / -128
  ;;
  ;; ================== L18 ==================
  ;; op_index          : 19
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 96 -> 96
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 28 x 28
  ;; depth_mult        : 1
  ;; w_off/b_off       : 10160 / 2656
  ;; mul_off/q6_off     : 2656 / 2656
  ;; wptr/bias/mul/q6   : 12208 / 1666752 / 1698912 / 1731072
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L19 ==================
  ;; op_index          : 20
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 0 -> 2
  ;; in_ptr/out_ptr    : 1767856 -> 2972080
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 96 -> 16
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 11024 / 3040
  ;; mul_off/q6_off     : 3040 / 3040
  ;; wptr/bias/mul/q6   : 13072 / 1667136 / 1699296 / 0
  ;; zx/zw/zy           : -128 / 0 / 0
  ;;
  ;; ================== L20 ==================
  ;; op_index          : 21
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [1 e 2] -> 0
  ;; in_ptr/out_ptr    : 2369968 -> 1767856
  ;; input_ptrs        : [2369968, 2972080]
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 16 -> 16
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 2369968 2972080 0 0 (input_ptrs)
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : 16 / 0 / 0
  ;;
  ;; ================== L21 ==================
  ;; op_index          : 22
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 16 -> 96
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 28 x 28
  ;; w_off/b_off       : 12560 / 3104
  ;; mul_off/q6_off     : 3104 / 3104
  ;; wptr/bias/mul/q6   : 14608 / 1667200 / 1699360 / 1731520
  ;; zx/zw/zy           : 0 / 0 / -128
  ;;
  ;; ================== L22 ==================
  ;; op_index          : 23
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 28 x 28
  ;; cin/cout          : 96 -> 96
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 2 x 2
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 1 0 1
  ;; out_h/out_w       : 14 x 14
  ;; depth_mult        : 1
  ;; w_off/b_off       : 14096 / 3488
  ;; mul_off/q6_off     : 3488 / 3488
  ;; wptr/bias/mul/q6   : 16144 / 1667584 / 1699744 / 1731904
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L23 ==================
  ;; op_index          : 24
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 96 -> 24
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 14960 / 3872
  ;; mul_off/q6_off     : 3872 / 3872
  ;; wptr/bias/mul/q6   : 17008 / 1667968 / 1700128 / 0
  ;; zx/zw/zy           : -128 / 0 / 4
  ;;
  ;; ================== L24 ==================
  ;; op_index          : 25
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 24 -> 144
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 17264 / 3968
  ;; mul_off/q6_off     : 3968 / 3968
  ;; wptr/bias/mul/q6   : 19312 / 1668064 / 1700224 / 1732384
  ;; zx/zw/zy           : 4 / 0 / -128
  ;;
  ;; ================== L25 ==================
  ;; op_index          : 26
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 144
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 14 x 14
  ;; depth_mult        : 1
  ;; w_off/b_off       : 20720 / 4544
  ;; mul_off/q6_off     : 4544 / 4544
  ;; wptr/bias/mul/q6   : 22768 / 1668640 / 1700800 / 1732960
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L26 ==================
  ;; op_index          : 27
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 1
  ;; in_ptr/out_ptr    : 2972080 -> 2369968
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 24
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 22016 / 5120
  ;; mul_off/q6_off     : 5120 / 5120
  ;; wptr/bias/mul/q6   : 24064 / 1669216 / 1701376 / 0
  ;; zx/zw/zy           : -128 / 0 / -4
  ;;
  ;; ================== L27 ==================
  ;; op_index          : 28
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [0 e 1] -> 2
  ;; in_ptr/out_ptr    : 1767856 -> 2972080
  ;; input_ptrs        : [1767856, 2369968]
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 24 -> 24
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1767856 2369968 0 0 (input_ptrs)
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : 4 / 0 / -4
  ;;
  ;; ================== L28 ==================
  ;; op_index          : 29
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 24 -> 144
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 25472 / 5216
  ;; mul_off/q6_off     : 5216 / 5216
  ;; wptr/bias/mul/q6   : 27520 / 1669312 / 1701472 / 1733632
  ;; zx/zw/zy           : -4 / 0 / -128
  ;;
  ;; ================== L29 ==================
  ;; op_index          : 30
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 144
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 14 x 14
  ;; depth_mult        : 1
  ;; w_off/b_off       : 28928 / 5792
  ;; mul_off/q6_off     : 5792 / 5792
  ;; wptr/bias/mul/q6   : 30976 / 1669888 / 1702048 / 1734208
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L30 ==================
  ;; op_index          : 31
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 1 -> 0
  ;; in_ptr/out_ptr    : 2369968 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 24
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 30224 / 6368
  ;; mul_off/q6_off     : 6368 / 6368
  ;; wptr/bias/mul/q6   : 32272 / 1670464 / 1702624 / 0
  ;; zx/zw/zy           : -128 / 0 / 9
  ;;
  ;; ================== L31 ==================
  ;; op_index          : 32
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [2 e 0] -> 1
  ;; in_ptr/out_ptr    : 2972080 -> 2369968
  ;; input_ptrs        : [2972080, 1767856]
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 24 -> 24
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 2972080 1767856 0 0 (input_ptrs)
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : -4 / 0 / 9
  ;;
  ;; ================== L32 ==================
  ;; op_index          : 33
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 24 -> 144
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 33680 / 6464
  ;; mul_off/q6_off     : 6464 / 6464
  ;; wptr/bias/mul/q6   : 35728 / 1670560 / 1702720 / 1734880
  ;; zx/zw/zy           : 9 / 0 / -128
  ;;
  ;; ================== L33 ==================
  ;; op_index          : 34
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 144
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 14 x 14
  ;; depth_mult        : 1
  ;; w_off/b_off       : 37136 / 7040
  ;; mul_off/q6_off     : 7040 / 7040
  ;; wptr/bias/mul/q6   : 39184 / 1671136 / 1703296 / 1735456
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L34 ==================
  ;; op_index          : 35
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 0 -> 2
  ;; in_ptr/out_ptr    : 1767856 -> 2972080
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 24
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 38432 / 7616
  ;; mul_off/q6_off     : 7616 / 7616
  ;; wptr/bias/mul/q6   : 40480 / 1671712 / 1703872 / 0
  ;; zx/zw/zy           : -128 / 0 / -16
  ;;
  ;; ================== L35 ==================
  ;; op_index          : 36
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [1 e 2] -> 0
  ;; in_ptr/out_ptr    : 2369968 -> 1767856
  ;; input_ptrs        : [2369968, 2972080]
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 24 -> 24
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 2369968 2972080 0 0 (input_ptrs)
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : 9 / 0 / -16
  ;;
  ;; ================== L36 ==================
  ;; op_index          : 37
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 24 -> 144
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 41888 / 7712
  ;; mul_off/q6_off     : 7712 / 7712
  ;; wptr/bias/mul/q6   : 43936 / 1671808 / 1703968 / 1736128
  ;; zx/zw/zy           : -16 / 0 / -128
  ;;
  ;; ================== L37 ==================
  ;; op_index          : 38
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 144
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 14 x 14
  ;; depth_mult        : 1
  ;; w_off/b_off       : 45344 / 8288
  ;; mul_off/q6_off     : 8288 / 8288
  ;; wptr/bias/mul/q6   : 47392 / 1672384 / 1704544 / 1736704
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L38 ==================
  ;; op_index          : 39
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 144 -> 32
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 46640 / 8864
  ;; mul_off/q6_off     : 8864 / 8864
  ;; wptr/bias/mul/q6   : 48688 / 1672960 / 1705120 / 0
  ;; zx/zw/zy           : -128 / 0 / -2
  ;;
  ;; ================== L39 ==================
  ;; op_index          : 40
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 32 -> 192
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 51248 / 8992
  ;; mul_off/q6_off     : 8992 / 8992
  ;; wptr/bias/mul/q6   : 53296 / 1673088 / 1705248 / 1737408
  ;; zx/zw/zy           : -2 / 0 / -128
  ;;
  ;; ================== L40 ==================
  ;; op_index          : 41
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 192 -> 192
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 14 x 14
  ;; depth_mult        : 1
  ;; w_off/b_off       : 57392 / 9760
  ;; mul_off/q6_off     : 9760 / 9760
  ;; wptr/bias/mul/q6   : 59440 / 1673856 / 1706016 / 1738176
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L41 ==================
  ;; op_index          : 42
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 1
  ;; in_ptr/out_ptr    : 2972080 -> 2369968
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 192 -> 32
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 59120 / 10528
  ;; mul_off/q6_off     : 10528 / 10528
  ;; wptr/bias/mul/q6   : 61168 / 1674624 / 1706784 / 0
  ;; zx/zw/zy           : -128 / 0 / 4
  ;;
  ;; ================== L42 ==================
  ;; op_index          : 43
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [0 e 1] -> 2
  ;; in_ptr/out_ptr    : 1767856 -> 2972080
  ;; input_ptrs        : [1767856, 2369968]
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 32 -> 32
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1767856 2369968 0 0 (input_ptrs)
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : -2 / 0 / 4
  ;;
  ;; ================== L43 ==================
  ;; op_index          : 44
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 32 -> 192
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 65264 / 10656
  ;; mul_off/q6_off     : 10656 / 10656
  ;; wptr/bias/mul/q6   : 67312 / 1674752 / 1706912 / 1739072
  ;; zx/zw/zy           : 4 / 0 / -128
  ;;
  ;; ================== L44 ==================
  ;; op_index          : 45
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 192 -> 192
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 14 x 14
  ;; depth_mult        : 1
  ;; w_off/b_off       : 71408 / 11424
  ;; mul_off/q6_off     : 11424 / 11424
  ;; wptr/bias/mul/q6   : 73456 / 1675520 / 1707680 / 1739840
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L45 ==================
  ;; op_index          : 46
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 1 -> 0
  ;; in_ptr/out_ptr    : 2369968 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 192 -> 32
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 73136 / 12192
  ;; mul_off/q6_off     : 12192 / 12192
  ;; wptr/bias/mul/q6   : 75184 / 1676288 / 1708448 / 0
  ;; zx/zw/zy           : -128 / 0 / 27
  ;;
  ;; ================== L46 ==================
  ;; op_index          : 47
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [2 e 0] -> 1
  ;; in_ptr/out_ptr    : 2972080 -> 2369968
  ;; input_ptrs        : [2972080, 1767856]
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 32 -> 32
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 2972080 1767856 0 0 (input_ptrs)
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : 4 / 0 / 27
  ;;
  ;; ================== L47 ==================
  ;; op_index          : 48
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 32 -> 192
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 14 x 14
  ;; w_off/b_off       : 79280 / 12320
  ;; mul_off/q6_off     : 12320 / 12320
  ;; wptr/bias/mul/q6   : 81328 / 1676416 / 1708576 / 1740736
  ;; zx/zw/zy           : 27 / 0 / -128
  ;;
  ;; ================== L48 ==================
  ;; op_index          : 49
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 14 x 14
  ;; cin/cout          : 192 -> 192
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 2 x 2
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 1 0 1
  ;; out_h/out_w       : 7 x 7
  ;; depth_mult        : 1
  ;; w_off/b_off       : 85424 / 13088
  ;; mul_off/q6_off     : 13088 / 13088
  ;; wptr/bias/mul/q6   : 87472 / 1677184 / 1709344 / 1741504
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L49 ==================
  ;; op_index          : 50
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 192 -> 56
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 87152 / 13856
  ;; mul_off/q6_off     : 13856 / 13856
  ;; wptr/bias/mul/q6   : 89200 / 1677952 / 1710112 / 0
  ;; zx/zw/zy           : -128 / 0 / 10
  ;;
  ;; ================== L50 ==================
  ;; op_index          : 51
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 56 -> 336
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 97904 / 14080
  ;; mul_off/q6_off     : 14080 / 14080
  ;; wptr/bias/mul/q6   : 99952 / 1678176 / 1710336 / 1742496
  ;; zx/zw/zy           : 10 / 0 / -128
  ;;
  ;; ================== L51 ==================
  ;; op_index          : 52
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 336 -> 336
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 7 x 7
  ;; depth_mult        : 1
  ;; w_off/b_off       : 116720 / 15424
  ;; mul_off/q6_off     : 15424 / 15424
  ;; wptr/bias/mul/q6   : 118768 / 1679520 / 1711680 / 1743840
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L52 ==================
  ;; op_index          : 53
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 0 -> 2
  ;; in_ptr/out_ptr    : 1767856 -> 2972080
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 336 -> 56
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 119744 / 16768
  ;; mul_off/q6_off     : 16768 / 16768
  ;; wptr/bias/mul/q6   : 121792 / 1680864 / 1713024 / 0
  ;; zx/zw/zy           : -128 / 0 / 16
  ;;
  ;; ================== L53 ==================
  ;; op_index          : 54
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [1 e 2] -> 0
  ;; in_ptr/out_ptr    : 2369968 -> 1767856
  ;; input_ptrs        : [2369968, 2972080]
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 56 -> 56
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 2369968 2972080 0 0 (input_ptrs)
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : 10 / 0 / 16
  ;;
  ;; ================== L54 ==================
  ;; op_index          : 55
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 56 -> 336
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 138560 / 16992
  ;; mul_off/q6_off     : 16992 / 16992
  ;; wptr/bias/mul/q6   : 140608 / 1681088 / 1713248 / 1745408
  ;; zx/zw/zy           : 16 / 0 / -128
  ;;
  ;; ================== L55 ==================
  ;; op_index          : 56
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 336 -> 336
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 7 x 7
  ;; depth_mult        : 1
  ;; w_off/b_off       : 157376 / 18336
  ;; mul_off/q6_off     : 18336 / 18336
  ;; wptr/bias/mul/q6   : 159424 / 1682432 / 1714592 / 1746752
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L56 ==================
  ;; op_index          : 57
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 2 -> 1
  ;; in_ptr/out_ptr    : 2972080 -> 2369968
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 336 -> 56
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 160400 / 19680
  ;; mul_off/q6_off     : 19680 / 19680
  ;; wptr/bias/mul/q6   : 162448 / 1683776 / 1715936 / 0
  ;; zx/zw/zy           : -128 / 0 / -9
  ;;
  ;; ================== L57 ==================
  ;; op_index          : 58
  ;; optype            : ADD
  ;; op_type           : 4 (ADD)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : [0 e 1] -> 2
  ;; in_ptr/out_ptr    : 1767856 -> 2972080
  ;; input_ptrs        : [1767856, 2369968]
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 56 -> 56
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1767856 2369968 0 0 (input_ptrs)
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : 16 / 0 / -9
  ;;
  ;; ================== L58 ==================
  ;; op_index          : 59
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 56 -> 336
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 179216 / 19904
  ;; mul_off/q6_off     : 19904 / 19904
  ;; wptr/bias/mul/q6   : 181264 / 1684000 / 1716160 / 1748320
  ;; zx/zw/zy           : -9 / 0 / -128
  ;;
  ;; ================== L59 ==================
  ;; op_index          : 60
  ;; optype            : DEPTHWISE_CONV_2D
  ;; op_type           : 2 (DW)
  ;; act               : 3 (RELU6)
  ;; flags             : 3 (PADDING_SAME|HAS_Q6)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 336 -> 336
  ;; kh/kw             : 3 x 3
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1 1 1 1
  ;; out_h/out_w       : 7 x 7
  ;; depth_mult        : 1
  ;; w_off/b_off       : 198032 / 21248
  ;; mul_off/q6_off     : 21248 / 21248
  ;; wptr/bias/mul/q6   : 200080 / 1685344 / 1717504 / 1749664
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L60 ==================
  ;; op_index          : 61
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 0 (NONE)
  ;; flags             : 1 (PADDING_SAME)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 336 -> 112
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 201056 / 22592
  ;; mul_off/q6_off     : 22592 / 22592
  ;; wptr/bias/mul/q6   : 203104 / 1686688 / 1718848 / 0
  ;; zx/zw/zy           : -128 / 0 / -21
  ;;
  ;; ================== L61 ==================
  ;; op_index          : 62
  ;; optype            : CONV_2D
  ;; op_type           : 1 (CONV)
  ;; act               : 3 (RELU6)
  ;; flags             : 2 (HAS_Q6)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 112 -> 1280
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 7 x 7
  ;; w_off/b_off       : 238688 / 23040
  ;; mul_off/q6_off     : 23040 / 23040
  ;; wptr/bias/mul/q6   : 240736 / 1687136 / 1719296 / 1751456
  ;; zx/zw/zy           : -21 / 0 / -128
  ;;
  ;; ================== L62 ==================
  ;; op_index          : 63
  ;; optype            : MEAN
  ;; op_type           : 5 (MEAN)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : 0 -> 1
  ;; in_ptr/out_ptr    : 1767856 -> 2369968
  ;; in_h/in_w         : 7 x 7
  ;; cin/cout          : 1280 -> 1280
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 1767856 0 0 0
  ;; out_h/out_w       : 1 x 1
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : -128 / 0 / -128
  ;;
  ;; ================== L63 ==================
  ;; op_index          : 64
  ;; optype            : FULLY_CONNECTED
  ;; op_type           : 3 (FC)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : 1 -> 2
  ;; in_ptr/out_ptr    : 2369968 -> 2972080
  ;; in_h/in_w         : 1 x 1
  ;; cin/cout          : 1280 -> 1000
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 0 0 0 0
  ;; out_h/out_w       : 1 x 1
  ;; w_off/b_off       : 382048 / 28160
  ;; mul_off/q6_off     : 28160 / 28160
  ;; wptr/bias/mul/q6   : 384096 / 1692256 / 1724416 / 0
  ;; zx/zw/zy           : -128 / 0 / -45
  ;;
  ;; ================== L64 ==================
  ;; op_index          : 65
  ;; optype            : SOFTMAX
  ;; op_type           : 6 (SOFTMAX)
  ;; act               : 0 (NONE)
  ;; flags             : 0 (0)
  ;; in_slot/out_slot  : 2 -> 0
  ;; in_ptr/out_ptr    : 2972080 -> 1767856
  ;; in_h/in_w         : 1 x 1
  ;; cin/cout          : 1000 -> 1000
  ;; kh/kw             : 1 x 1
  ;; stride_h/stride_w : 1 x 1
  ;; dil_h/dil_w       : 1 x 1
  ;; pad t/b/l/r       : 2972080 0 0 0
  ;; out_h/out_w       : 1 x 1
  ;; w_off/b_off       : 0 / 0
  ;; mul_off/q6_off     : 0 / 0
  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
  ;; zx/zw/zy           : -45 / 0 / -128
  ;;

  ;; ============================================================
  ;; LayerParam reader (28x i32) + loader to locals
  ;; ============================================================

  (global $LP_SIZE i32 (i32.const 112)) ;; 28 * 4

  ;; Field indices (28 i32):
  ;;  0 op_type
  ;;  1 act
  ;;  2 flags
  ;;  3 in_ptr
  ;;  4 out_ptr
  ;;  5 in_h
  ;;  6 in_w
  ;;  7 cin
  ;;  8 cout
  ;;  9 kh
  ;; 10 kw
  ;; 11 stride_h
  ;; 12 stride_w
  ;; 13 dil_h
  ;; 14 dil_w
  ;; 15 pad_t
  ;; 16 pad_b
  ;; 17 pad_l
  ;; 18 pad_r
  ;; 19 wptr
  ;; 20 bias_ptr
  ;; 21 mul_ptr
  ;; 22 q6_ptr
  ;; 23 zx
  ;; 24 zw
  ;; 25 zy
  ;; 26 out_h
  ;; 27 out_w

  (func $layerparam_base (param $layer_idx i32) (result i32)
    global.get $PARAMS_BASE
    local.get $layer_idx
    global.get $LP_SIZE
    i32.mul
    i32.add
  )

  ;; Helpers para quantização Q31
  (func $saturating_rounding_doubling_high_mul (param $a i32) (param $b i32) (result i32)
    (local $ab i64)
    (local $nudge i64)
    (local $result i64)
    
    ;; Caso especial: INT32_MIN * INT32_MIN
    local.get $a
    i32.const -2147483648
    i32.eq
    local.get $b
    i32.const -2147483648
    i32.eq
    i32.and
    (if
      (then
        i32.const 2147483647
        return
      )
    )
    
    ;; ab = a * b
    local.get $a
    i64.extend_i32_s
    local.get $b
    i64.extend_i32_s
    i64.mul
    local.set $ab
    
    ;; nudge = (ab >= 0) ? (1 << 30) : (1 - (1 << 30))
    local.get $ab
    i64.const 0
    i64.ge_s
    (if
      (then
        i64.const 1073741824  ;; 1 << 30
        local.set $nudge
      )
      (else
        i64.const -1073741823  ;; 1 - (1 << 30)
        local.set $nudge
      )
    )
    
    ;; result = (ab + nudge) >> 31
    local.get $ab
    local.get $nudge
    i64.add
    i64.const 31
    i64.shr_s
    local.set $result
    
    ;; Saturate
    local.get $result
    i64.const 2147483647
    i64.gt_s
    (if
      (then
        i32.const 2147483647
        return
      )
    )
    
    local.get $result
    i64.const -2147483648
    i64.lt_s
    (if
      (then
        i32.const -2147483648
        return
      )
    )
    
    local.get $result
    i32.wrap_i64
  )

  (func $rounding_divide_by_pot (param $x i32) (param $exponent i32) (result i32)
    (local $mask i32)
    (local $remainder i32)
    (local $threshold i32)
    (local $base i32)
    
    ;; Se exponent <= 0, retorna x
    local.get $exponent
    i32.const 0
    i32.le_s
    (if
      (then
        local.get $x
        return
      )
    )
    
    ;; mask = (1 << exponent) - 1
    i32.const 1
    local.get $exponent
    i32.shl
    i32.const 1
    i32.sub
    local.set $mask
    
    ;; remainder = x & mask
    local.get $x
    local.get $mask
    i32.and
    local.set $remainder
    
    ;; threshold = mask >> 1
    local.get $mask
    i32.const 1
    i32.shr_u
    local.set $threshold
    
    ;; Se x < 0, threshold++
    local.get $x
    i32.const 0
    i32.lt_s
    (if
      (then
        local.get $threshold
        i32.const 1
        i32.add
        local.set $threshold
      )
    )

    ;; base = x >> exponent
    local.get $x
    local.get $exponent
    i32.shr_s
    local.set $base
    
    ;; return base + (remainder > threshold ? 1 : 0)
    local.get $base
    local.get $remainder
    local.get $threshold
    i32.gt_s
    (if (result i32)
      (then
        i32.const 1
      )
      (else
        i32.const 0
      )
    )
    i32.add
  )

  (func $multiply_by_quantized_multiplier (param $x i32) (param $multiplier i32) (param $shift i32) (result i32)
    (local $result i32)
    
    local.get $x
    local.set $result
    
    ;; Se shift > 0: left shift antes do mul
    local.get $shift
    i32.const 0
    i32.gt_s
    (if
      (then
        local.get $result
        local.get $shift
        i32.shl
        local.set $result
      )
    )
    
    ;; Multiplicação Q31
    local.get $result
    local.get $multiplier
    call $saturating_rounding_doubling_high_mul
    local.set $result
    
    ;; Se shift < 0: right shift depois do mul
    local.get $shift
    i32.const 0
    i32.lt_s
    (if
      (then
        local.get $result
        local.get $shift
        i32.const 0
        i32.sub  ;; -shift
        call $rounding_divide_by_pot
        local.set $result
      )
    )
    
    local.get $result
  )

  ;; helpers inlined inside each function:
  ;; expand5to8(x) = (x<<3) | (x>>2)
  ;; expand6to8(x) = (x<<2) | (x>>4)

  (func $red_from_rgb565 (export "red_from_rgb565") (param $v i32) (result i32)
    (local $r5 i32)
    (local $r8 i32)

    ;; r5 = (v >>> 3) & 31
    local.get $v
    i32.const 3
    i32.shr_u
    i32.const 31
    i32.and
    local.set $r5

    ;; r8 = (r5 << 3) | (r5 >>> 2)
    local.get $r5
    i32.const 3
    i32.shl
    local.get $r5
    i32.const 2
    i32.shr_u
    i32.or
    local.set $r8

    ;; return r8 - 128
    local.get $r8
    i32.const 128
    i32.sub
  )

  (func $blue_from_rgb565 (export "blue_from_rgb565") (param $v i32) (result i32)
    (local $b5 i32)
    (local $b8 i32)

    ;; b5 = (v >>> 8) & 31
    local.get $v
    i32.const 8
    i32.shr_u
    i32.const 31
    i32.and
    local.set $b5

    ;; b8 = (b5 << 3) | (b5 >>> 2)
    local.get $b5
    i32.const 3
    i32.shl
    local.get $b5
    i32.const 2
    i32.shr_u
    i32.or
    local.set $b8

    ;; return b8 - 128
    local.get $b8
    i32.const 128
    i32.sub
  )

  (func $green_from_rgb565 (export "green_from_rgb565") (param $v i32) (result i32)
    (local $msb i32)
    (local $lsb i32)
    (local $g6  i32)
    (local $g8  i32)

    ;; msb = (v >>> 13) & 7
    local.get $v
    i32.const 13
    i32.shr_u
    i32.const 7
    i32.and
    local.set $msb

    ;; lsb = (v & 7) << 3
    local.get $v
    i32.const 7
    i32.and
    i32.const 3
    i32.shl
    local.set $lsb

    ;; g6 = (msb | lsb) & 63
    local.get $msb
    local.get $lsb
    i32.or
    i32.const 63
    i32.and
    local.set $g6

    ;; g8 = (g6 << 2) | (g6 >>> 4)
    local.get $g6
    i32.const 2
    i32.shl
    local.get $g6
    i32.const 4
    i32.shr_u
    i32.or
    local.set $g8

    ;; return g8 - 128
    local.get $g8
    i32.const 128
    i32.sub
  )


  (func $conv2d_layer0 (export "conv2d_layer0") (param $layer_idx i32)
    (local $base     i32)

    (local $op_type   i32)
    (local $act       i32)
    (local $flags     i32)
    (local $in_ptr    i32)
    (local $out_ptr   i32)
    (local $in_h      i32)
    (local $in_w      i32)
    (local $cin       i32)
    (local $cout      i32)
    (local $kh        i32)
    (local $kw        i32)
    (local $stride_h  i32)
    (local $stride_w  i32)
    (local $dil_h     i32)
    (local $dil_w     i32)
    (local $pad_t     i32)
    (local $pad_b     i32)
    (local $pad_l     i32)
    (local $pad_r     i32)
    (local $wptr      i32)
    (local $bias_ptr  i32)
    (local $mul_ptr   i32)
    (local $q6_ptr    i32)
    (local $zx        i32)
    (local $zw        i32)
    (local $zy        i32)
    (local $out_h     i32)
    (local $out_w     i32)

    (local $bottom    i32)
    (local $right     i32)
    (local $plane_out i32)
    (local $w_per_oc  i32)

    (local $oc        i32)
    (local $i_out     i32)
    (local $j_out     i32)
    (local $ki        i32)
    (local $kj        i32)

    (local $baseOutOC i32)
    (local $baseWoc   i32)

    (local $b         i32)
    (local $m         i32)
    (local $q6        i32)

    (local $i         i32)
    (local $j         i32)

    (local $acc       i32)

    (local $row       i32)
    (local $col       i32)

    (local $row_img   i32)
    (local $row_base  i32)
    (local $col_img   i32)
    (local $idx       i32)

    (local $pix16     i32)
    (local $r         i32)
    (local $g         i32)
    (local $bch       i32)

    (local $pos       i32)
    (local $baseW_kikj i32)

    (local $kR        i32)
    (local $kG        i32)
    (local $kB        i32)

    (local $outIdx    i32)

    (local $tmp i32)
    (local $w0  i32)

    (local $y   i32)
    (local $hi  i32)
    (local $lo  i32)
    (local $p   i64)
    (local $nudge i64)

    ;; base = PARAMS_BASE + layer_idx * LP_SIZE
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    ;; -------------------------
    ;; helpers inline:
    ;; load_i32(field_idx) = i32.load(base + field_idx*4)
    ;; -------------------------

    ;; op_type (0)
    local.get $base
    i32.const 0
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $op_type

    ;; act (1)
    local.get $base
    i32.const 1
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $act

    ;; flags (2)
    local.get $base
    i32.const 2
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $flags

    ;; in_ptr (3)
    local.get $base
    i32.const 3
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr

    ;; out_ptr (4)
    local.get $base
    i32.const 4
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_ptr

    ;; in_h (5)
    local.get $base
    i32.const 5
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_h

    ;; in_w (6)
    local.get $base
    i32.const 6
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_w

    ;; cin (7)
    local.get $base
    i32.const 7
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cin

    ;; cout (8)
    local.get $base
    i32.const 8
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cout

    ;; kh (9)
    local.get $base
    i32.const 9
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $kh

    ;; kw (10)
    local.get $base
    i32.const 10
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $kw

    ;; stride_h (11)
    local.get $base
    i32.const 11
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $stride_h

    ;; stride_w (12)
    local.get $base
    i32.const 12
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $stride_w

    ;; dil_h (13)
    local.get $base
    i32.const 13
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $dil_h

    ;; dil_w (14)
    local.get $base
    i32.const 14
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $dil_w

    ;; pad_t (15)
    local.get $base
    i32.const 15
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_t

    ;; pad_b (16)
    local.get $base
    i32.const 16
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_b

    ;; pad_l (17)
    local.get $base
    i32.const 17
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_l

    ;; pad_r (18)
    local.get $base
    i32.const 18
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_r

    ;; wptr (19)
    local.get $base
    i32.const 19
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $wptr

    ;; bias_ptr (20)
    local.get $base
    i32.const 20
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $bias_ptr

    ;; mul_ptr (21)
    local.get $base
    i32.const 21
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $mul_ptr

    ;; q6_ptr (22)
    local.get $base
    i32.const 22
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $q6_ptr

    ;; zx (23)
    local.get $base
    i32.const 23
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zx

    ;; zw (24)
    local.get $base
    i32.const 24
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zw

    ;; zy (25)
    local.get $base
    i32.const 25
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zy

    ;; out_h (26)
    local.get $base
    i32.const 26
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_h

    ;; out_w (27)
    local.get $base
    i32.const 27
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_w

    ;; =========================


    ;; bottom = pad_t + in_h;
    local.get $pad_t
    local.get $in_h
    i32.add
    local.set $bottom

    ;; right = pad_l + in_w;
    local.get $pad_l
    local.get $in_w
    i32.add
    local.set $right

    ;; plane_out = out_h * out_w;
    local.get $out_h
    local.get $out_w
    i32.mul
    local.set $plane_out

    ;; // bytes por oc: kh*kw*cin = 9*3 = 27
    local.get $kh
    local.get $kw
    local.get $cin
    i32.mul
    i32.mul
    local.set $w_per_oc

    ;; =========================
    ;; loops: oc / i_out / j_out / ki / kj
    ;; com dilatação: row = i + ki*dil_h, col = j + kj*dil_w
    ;; =========================

    ;; oc = 0
    i32.const 0
    local.set $oc

    (block $exit_oc
      (loop $loop_oc

        ;; if (oc >= cout) break;
        local.get $oc
        local.get $cout
        i32.ge_s
        br_if $exit_oc

        ;; baseOutOC = out_ptr + oc * plane_out
        local.get $out_ptr
        local.get $oc
        local.get $plane_out
        i32.mul
        i32.add
        local.set $baseOutOC

        ;; b = load<i32>(bias_ptr + oc*4)
        local.get $bias_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $b

        ;; m = load<i32>(mul_ptr + oc*4)
        local.get $mul_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $m

        ;; q6 = (q6_ptr != 0) ? load<i32>(q6_ptr + oc*4) : 0
        local.get $q6_ptr
        i32.eqz
        (if (result i32)
          (then
            i32.const 0
          )
          (else
            local.get $q6_ptr
            local.get $oc
            i32.const 2
            i32.shl
            i32.add
            i32.load align=4
          )
        )
        local.set $q6

        ;; baseWoc = wptr + oc * w_per_oc
        local.get $wptr
        local.get $oc
        local.get $w_per_oc
        i32.mul
        i32.add
        local.set $baseWoc

        ;; i_out = 0
        i32.const 0
        local.set $i_out

        (block $exit_i
          (loop $loop_i

            ;; if (i_out >= out_h) break;
            local.get $i_out
            local.get $out_h
            i32.ge_s
            br_if $exit_i

            ;; i = i_out * stride_h
            local.get $i_out
            local.get $stride_h
            i32.mul
            local.set $i

            ;; j_out = 0
            i32.const 0
            local.set $j_out

            (block $exit_j
              (loop $loop_j

                ;; if (j_out >= out_w) break;
                local.get $j_out
                local.get $out_w
                i32.ge_s
                br_if $exit_j

                ;; j = j_out * stride_w
                local.get $j_out
                local.get $stride_w
                i32.mul
                local.set $j

                ;; acc = b
                local.get $b
                local.set $acc

                ;; ki = 0
                i32.const 0
                local.set $ki

                (block $exit_ki
                  (loop $loop_ki

                    ;; if (ki >= kh) break;
                    local.get $ki
                    local.get $kh
                    i32.ge_s
                    br_if $exit_ki

                    ;; ---- alvo do "continue" do ki ----
                    (block $inc_ki

                      ;; row = i + ki*dil_h   (DILATAÇÃO AQUI)
                      local.get $i
                      local.get $ki
                      local.get $dil_h
                      i32.mul
                      i32.add
                      local.set $row

                      ;; if (row < pad_t) continue;
                      local.get $row
                      local.get $pad_t
                      i32.lt_s
                      br_if $inc_ki

                      ;; if (row >= bottom) break;
                      local.get $row
                      local.get $bottom
                      i32.ge_s
                      br_if $exit_ki

                      ;; row_img  = row - pad_t
                      local.get $row
                      local.get $pad_t
                      i32.sub
                      local.set $row_img

                      ;; row_base = row_img * in_w
                      local.get $row_img
                      local.get $in_w
                      i32.mul
                      local.set $row_base

                      ;; kj = 0
                      i32.const 0
                      local.set $kj

                      (block $exit_kj
                        (loop $loop_kj

                          ;; if (kj >= kw) break;
                          local.get $kj
                          local.get $kw
                          i32.ge_s
                          br_if $exit_kj

                          ;; ---- alvo do "continue" do kj ----
                          (block $inc_kj

                            ;; col = j + kj*dil_w   (DILATAÇÃO AQUI)
                            local.get $j
                            local.get $kj
                            local.get $dil_w
                            i32.mul
                            i32.add
                            local.set $col

                            ;; if (col < pad_l) continue;
                            local.get $col
                            local.get $pad_l
                            i32.lt_s
                            br_if $inc_kj

                            ;; if (col >= right) break;
                            local.get $col
                            local.get $right
                            i32.ge_s
                            br_if $exit_kj

                            ;; col_img = col - pad_l
                            local.get $col
                            local.get $pad_l
                            i32.sub
                            local.set $col_img

                            ;; idx = row_base + col_img
                            local.get $row_base
                            local.get $col_img
                            i32.add
                            local.set $idx

                            ;; pix16 = load<u16>(in_ptr + (idx<<1))
                            local.get $in_ptr
                            local.get $idx
                            i32.const 1
                            i32.shl
                            i32.add
                            i32.load16_u
                            local.set $pix16

                            ;; r,g,bch extração (mesmo mapping do seu WAT)
                            local.get $pix16
                            call $red_from_rgb565
                            local.set $r

                            local.get $pix16
                            call $green_from_rgb565
                            local.set $g

                            local.get $pix16
                            call $blue_from_rgb565
                            local.set $bch

                            ;; pos = (ki*kw + kj) * cin
                            local.get $ki
                            local.get $kw
                            i32.mul
                            local.get $kj
                            i32.add
                            local.get $cin
                            i32.mul
                            local.set $pos

                            ;; baseW_kikj = baseWoc + pos
                            local.get $baseWoc
                            local.get $pos
                            i32.add
                            local.set $baseW_kikj

                            ;; kR = load<i8>(baseW_kikj + 0)
                            local.get $baseW_kikj
                            i32.load8_s
                            local.set $kR

                            ;; kG = load<i8>(baseW_kikj + 1)
                            local.get $baseW_kikj
                            i32.const 1
                            i32.add
                            i32.load8_s
                            local.set $kG

                            ;; kB = load<i8>(baseW_kikj + 2)
                            local.get $baseW_kikj
                            i32.const 2
                            i32.add
                            i32.load8_s
                            local.set $kB

                            ;; -----------------------------------------
                            ;; acc += (r - zx) * (kR - zw)
                            ;; -----------------------------------------

                            ;; w0 = kR - zw
                            local.get $kR
                            local.get $zw
                            i32.sub
                            local.set $w0

                            ;; tmp = r - zx
                            local.get $r
                            local.get $zx
                            i32.sub
                            local.set $tmp

                            ;; acc = acc + tmp*w0
                            local.get $acc
                            local.get $tmp
                            local.get $w0
                            i32.mul
                            i32.add
                            local.set $acc

                            ;; -----------------------------------------
                            ;; acc += (g - zx) * (kG - zw)
                            ;; -----------------------------------------

                            ;; w0 = kG - zw
                            local.get $kG
                            local.get $zw
                            i32.sub
                            local.set $w0

                            ;; tmp = g - zx
                            local.get $g
                            local.get $zx
                            i32.sub
                            local.set $tmp

                            ;; acc = acc + tmp*w0
                            local.get $acc
                            local.get $tmp
                            local.get $w0
                            i32.mul
                            i32.add
                            local.set $acc

                            ;; -----------------------------------------
                            ;; acc += (bch - zx) * (kB - zw)
                            ;; -----------------------------------------

                            ;; w0 = kB - zw
                            local.get $kB
                            local.get $zw
                            i32.sub
                            local.set $w0

                            ;; tmp = bch - zx
                            local.get $bch
                            local.get $zx
                            i32.sub
                            local.set $tmp

                            ;; acc = acc + tmp*w0
                            local.get $acc
                            local.get $tmp
                            local.get $w0
                            i32.mul
                            i32.add
                            local.set $acc


                            ;; kj++
                            local.get $kj
                            i32.const 1
                            i32.add
                            local.set $kj

                            br $loop_kj
                          )

                          ;; kj++
                          local.get $kj
                          i32.const 1
                          i32.add
                          local.set $kj
                          br $loop_kj
                        )
                      )
                    )

                    ;; ki++
                    local.get $ki
                    i32.const 1
                    i32.add
                    local.set $ki
                    br $loop_ki
                  )
                )

                ;; outIdx = i_out*out_w + j_out
                local.get $i_out
                local.get $out_w
                i32.mul
                local.get $j_out
                i32.add
                local.set $outIdx

                ;; =========================
                ;; REQUANT + zp_y + activation + clamp + store8
                ;; =========================

                ;; p = (i64)acc * (i64)m
                local.get $acc
                i64.extend_i32_s
                local.get $m
                i64.extend_i32_s
                i64.mul
                local.set $p

                ;; RSHIFT = 31
                ;; nudge = 1 << (31-1) = 1<<30
                i64.const 1
                i64.const 30
                i64.shl
                local.set $nudge

                ;; if (p < 0) p -= nudge else p += nudge   (rounding)
                local.get $p
                i64.const 0
                i64.lt_s
                (if
                  (then
                    local.get $p
                    local.get $nudge
                    i64.sub
                    local.set $p
                  )
                  (else
                    local.get $p
                    local.get $nudge
                    i64.add
                    local.set $p
                  )
                )

                ;; y = (i32)(p >> 31)
                local.get $p
                i64.const 31
                i64.shr_s
                i32.wrap_i64
                local.set $y

                ;; y += zy   (zp_y)
                local.get $y
                local.get $zy
                i32.add
                local.set $y

                ;; -------- activation --------
                ;; act == RELU (1): y = max(y, zy)
                local.get $act
                i32.const 1
                i32.eq
                (if
                  (then
                    local.get $y
                    local.get $zy
                    i32.lt_s
                    (if
                      (then
                        local.get $zy
                        local.set $y
                      )
                    )
                  )
                )

                ;; act == RELU6 (3): clamp y to [zy, q6_abs]
                local.get $act
                i32.const 3
                i32.eq
                (if
                  (then
                    ;; lo = zy
                    local.get $zy
                    local.set $lo

                    ;; hi = q6   (já é q6_abs = round(6/sy) + zy)
                    local.get $q6
                    local.set $hi

                    ;; hi = min(hi, 127)
                    local.get $hi
                    i32.const 127
                    i32.gt_s
                    (if
                      (then
                        i32.const 127
                        local.set $hi
                      )
                    )

                    ;; lo = max(lo, -128)
                    local.get $lo
                    i32.const -128
                    i32.lt_s
                    (if
                      (then
                        i32.const -128
                        local.set $lo
                      )
                    )

                    ;; if (y < lo) y = lo
                    local.get $y
                    local.get $lo
                    i32.lt_s
                    (if
                      (then
                        local.get $lo
                        local.set $y
                      )
                    )

                    ;; if (y > hi) y = hi
                    local.get $y
                    local.get $hi
                    i32.gt_s
                    (if
                      (then
                        local.get $hi
                        local.set $y
                      )
                    )
                  )
                )


                ;; -------- final clamp [-128..127] --------
                local.get $y
                i32.const 127
                i32.gt_s
                (if
                  (then
                    i32.const 127
                    local.set $y
                  )
                )

                local.get $y
                i32.const -128
                i32.lt_s
                (if
                  (then
                    i32.const -128
                    local.set $y
                  )
                )

                ;; store8(out_ptr[oc][outIdx]) = (i8)y
                local.get $baseOutOC
                local.get $outIdx
                i32.add
                local.get $y
                i32.store8


                ;; j_out++
                local.get $j_out
                i32.const 1
                i32.add
                local.set $j_out

                br $loop_j
              )
            )

            ;; i_out++
            local.get $i_out
            i32.const 1
            i32.add
            local.set $i_out

            br $loop_i
          )
        )

        ;; oc++
        local.get $oc
        i32.const 1
        i32.add
        local.set $oc

        br $loop_oc
      )
    )
  )

  (func $depthwise_conv2d (export "depthwise_conv2d") (param $layer_idx i32)
    (local $base     i32)

    (local $op_type   i32)
    (local $act       i32)
    (local $flags     i32)
    (local $in_ptr    i32)
    (local $out_ptr   i32)
    (local $in_h      i32)
    (local $in_w      i32)
    (local $cin       i32)
    (local $cout      i32)
    (local $kh        i32)
    (local $kw        i32)
    (local $stride_h  i32)
    (local $stride_w  i32)
    (local $dil_h     i32)
    (local $dil_w     i32)
    (local $pad_t     i32)
    (local $pad_b     i32)
    (local $pad_l     i32)
    (local $pad_r     i32)
    (local $wptr      i32)
    (local $bias_ptr  i32)
    (local $mul_ptr   i32)
    (local $q6_ptr    i32)
    (local $zx        i32)
    (local $zw        i32)
    (local $zy        i32)
    (local $out_h     i32)
    (local $out_w     i32)

    (local $bottom    i32)
    (local $right     i32)
    (local $plane_out i32)
    (local $w_per_oc  i32)

    (local $oc        i32)
    (local $i_out     i32)
    (local $j_out     i32)
    (local $ki        i32)
    (local $kj        i32)

    (local $baseOutOC i32)
    (local $baseWoc   i32)

    (local $b         i32)
    (local $m         i32)
    (local $q6        i32)

    (local $i         i32)
    (local $j         i32)

    (local $acc       i32)

    (local $row       i32)
    (local $col       i32)

    (local $row_img   i32)
    (local $row_base  i32)
    (local $col_img   i32)
    (local $idx       i32)

    (local $pos       i32)
    (local $baseW_kikj i32)

    (local $outIdx    i32)

    (local $tmp i32)
    (local $w0  i32)

    (local $y   i32)
    (local $hi  i32)
    (local $lo  i32)
    (local $p   i64)
    (local $nudge i64)

    ;; base = PARAMS_BASE + layer_idx * LP_SIZE
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    ;; -------------------------
    ;; helpers inline:
    ;; load_i32(field_idx) = i32.load(base + field_idx*4)
    ;; -------------------------

    ;; op_type (0)
    local.get $base
    i32.const 0
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $op_type

    ;; act (1)
    local.get $base
    i32.const 1
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $act

    ;; flags (2)
    local.get $base
    i32.const 2
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $flags

    ;; in_ptr (3)
    local.get $base
    i32.const 3
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr

    ;; out_ptr (4)
    local.get $base
    i32.const 4
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_ptr

    ;; in_h (5)
    local.get $base
    i32.const 5
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_h

    ;; in_w (6)
    local.get $base
    i32.const 6
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_w

    ;; cin (7)
    local.get $base
    i32.const 7
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cin

    ;; cout (8)
    local.get $base
    i32.const 8
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cout

    ;; kh (9)
    local.get $base
    i32.const 9
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $kh

    ;; kw (10)
    local.get $base
    i32.const 10
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $kw

    ;; stride_h (11)
    local.get $base
    i32.const 11
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $stride_h

    ;; stride_w (12)
    local.get $base
    i32.const 12
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $stride_w

    ;; dil_h (13)
    local.get $base
    i32.const 13
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $dil_h

    ;; dil_w (14)
    local.get $base
    i32.const 14
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $dil_w

    ;; pad_t (15)
    local.get $base
    i32.const 15
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_t

    ;; pad_b (16)
    local.get $base
    i32.const 16
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_b

    ;; pad_l (17)
    local.get $base
    i32.const 17
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_l

    ;; pad_r (18)
    local.get $base
    i32.const 18
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_r

    ;; wptr (19)
    local.get $base
    i32.const 19
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $wptr

    ;; bias_ptr (20)
    local.get $base
    i32.const 20
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $bias_ptr

    ;; mul_ptr (21)
    local.get $base
    i32.const 21
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $mul_ptr

    ;; q6_ptr (22)
    local.get $base
    i32.const 22
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $q6_ptr

    ;; zx (23)
    local.get $base
    i32.const 23
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zx

    ;; zw (24)
    local.get $base
    i32.const 24
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zw

    ;; zy (25)
    local.get $base
    i32.const 25
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zy

    ;; out_h (26)
    local.get $base
    i32.const 26
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_h

    ;; out_w (27)
    local.get $base
    i32.const 27
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_w

    ;; =========================


    ;; bottom = pad_t + in_h;
    local.get $pad_t
    local.get $in_h
    i32.add
    local.set $bottom

    ;; right = pad_l + in_w;
    local.get $pad_l
    local.get $in_w
    i32.add
    local.set $right

    ;; plane_out = out_h * out_w;
    local.get $out_h
    local.get $out_w
    i32.mul
    local.set $plane_out

    ;; bytes por oc (depthwise): kh * kw = 9
    local.get $kh
    local.get $kw
    i32.mul
    local.set $w_per_oc

    ;; =========================
    ;; loops: oc / i_out / j_out / ki / kj
    ;; com dilatação: row = i + ki*dil_h, col = j + kj*dil_w
    ;; =========================

    ;; oc = 0
    i32.const 0
    local.set $oc

    (block $exit_oc
      (loop $loop_oc

        ;; if (oc >= cout) break;
        local.get $oc
        local.get $cout
        i32.ge_s
        br_if $exit_oc

        ;; baseOutOC = out_ptr + oc * plane_out
        local.get $out_ptr
        local.get $oc
        local.get $plane_out
        i32.mul
        i32.add
        local.set $baseOutOC

        ;; b = load<i32>(bias_ptr + oc*4)
        local.get $bias_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $b

        ;; m = load<i32>(mul_ptr + oc*4)
        local.get $mul_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $m

        ;; q6 = (q6_ptr != 0) ? load<i32>(q6_ptr + oc*4) : 0
        local.get $q6_ptr
        i32.eqz
        (if (result i32)
          (then
            i32.const 0
          )
          (else
            local.get $q6_ptr
            local.get $oc
            i32.const 2
            i32.shl
            i32.add
            i32.load align=4
          )
        )
        local.set $q6

        ;; baseWoc = wptr + oc * w_per_oc
        local.get $wptr
        local.get $oc
        local.get $w_per_oc
        i32.mul
        i32.add
        local.set $baseWoc

        ;; i_out = 0
        i32.const 0
        local.set $i_out

        (block $exit_i
          (loop $loop_i

            ;; if (i_out >= out_h) break;
            local.get $i_out
            local.get $out_h
            i32.ge_s
            br_if $exit_i

            ;; i = i_out * stride_h
            local.get $i_out
            local.get $stride_h
            i32.mul
            local.set $i

            ;; j_out = 0
            i32.const 0
            local.set $j_out

            (block $exit_j
              (loop $loop_j

                ;; if (j_out >= out_w) break;
                local.get $j_out
                local.get $out_w
                i32.ge_s
                br_if $exit_j

                ;; j = j_out * stride_w
                local.get $j_out
                local.get $stride_w
                i32.mul
                local.set $j

                ;; acc = b
                local.get $b
                local.set $acc

                ;; ki = 0
                i32.const 0
                local.set $ki

                (block $exit_ki
                  (loop $loop_ki

                    ;; if (ki >= kh) break;
                    local.get $ki
                    local.get $kh
                    i32.ge_s
                    br_if $exit_ki

                    ;; ---- alvo do "continue" do ki ----
                    (block $inc_ki

                      ;; row = i + ki*dil_h   (DILATAÇÃO AQUI)
                      local.get $i
                      local.get $ki
                      local.get $dil_h
                      i32.mul
                      i32.add
                      local.set $row

                      ;; if (row < pad_t) continue;
                      local.get $row
                      local.get $pad_t
                      i32.lt_s
                      br_if $inc_ki

                      ;; if (row >= bottom) break;
                      local.get $row
                      local.get $bottom
                      i32.ge_s
                      br_if $exit_ki

                      ;; row_img  = row - pad_t
                      local.get $row
                      local.get $pad_t
                      i32.sub
                      local.set $row_img

                      ;; row_base = row_img * in_w
                      local.get $row_img
                      local.get $in_w
                      i32.mul
                      local.set $row_base

                      ;; kj = 0
                      i32.const 0
                      local.set $kj

                      (block $exit_kj
                        (loop $loop_kj

                          ;; if (kj >= kw) break;
                          local.get $kj
                          local.get $kw
                          i32.ge_s
                          br_if $exit_kj

                          ;; ---- alvo do "continue" do kj ----
                          (block $inc_kj

                            ;; col = j + kj*dil_w   (DILATAÇÃO AQUI)
                            local.get $j
                            local.get $kj
                            local.get $dil_w
                            i32.mul
                            i32.add
                            local.set $col

                            ;; if (col < pad_l) continue;
                            local.get $col
                            local.get $pad_l
                            i32.lt_s
                            br_if $inc_kj

                            ;; if (col >= right) break;
                            local.get $col
                            local.get $right
                            i32.ge_s
                            br_if $exit_kj

                            ;; col_img = col - pad_l
                            local.get $col
                            local.get $pad_l
                            i32.sub
                            local.set $col_img

                            ;; -------------------------------------------------
                            ;; idx = ((row_img * in_w + col_img) * cin) + oc
                            ;; -------------------------------------------------
                            local.get $row_base
                            local.get $col_img
                            i32.add
                            local.get $cin
                            i32.mul
                            local.get $oc
                            i32.add
                            local.set $idx

                            ;; input = load<i8>(in_ptr + idx)
                            local.get $in_ptr
                            local.get $idx
                            i32.add
                            i32.load8_s
                            local.set $tmp

                            ;; -------------------------------------------------
                            ;; kernel index = oc*(kh*kw) + ki*kw + kj
                            ;; -------------------------------------------------
                            local.get $baseWoc
                            local.get $ki
                            local.get $kw
                            i32.mul
                            local.get $kj
                            i32.add
                            i32.add
                            i32.load8_s
                            local.set $w0

                            ;; -----------------------------------------
                            ;; acc += (input - zx) * (w - zw)
                            ;; -----------------------------------------
                            local.get $tmp
                            local.get $zx
                            i32.sub
                            local.get $w0
                            local.get $zw
                            i32.sub
                            i32.mul
                            local.get $acc
                            i32.add
                            local.set $acc

                            ;; kj++
                            local.get $kj
                            i32.const 1
                            i32.add
                            local.set $kj

                            br $loop_kj
                          )

                          ;; kj++
                          local.get $kj
                          i32.const 1
                          i32.add
                          local.set $kj
                          br $loop_kj
                        )
                      )
                    )

                    ;; ki++
                    local.get $ki
                    i32.const 1
                    i32.add
                    local.set $ki
                    br $loop_ki
                  )
                )

                ;; outIdx = i_out*out_w + j_out
                local.get $i_out
                local.get $out_w
                i32.mul
                local.get $j_out
                i32.add
                local.set $outIdx

                ;; =========================
                ;; REQUANT + zp_y + activation + clamp + store8
                ;; =========================

                ;; p = (i64)acc * (i64)m
                local.get $acc
                i64.extend_i32_s
                local.get $m
                i64.extend_i32_s
                i64.mul
                local.set $p

                ;; RSHIFT = 31
                ;; nudge = 1 << (31-1) = 1<<30
                i64.const 1
                i64.const 30
                i64.shl
                local.set $nudge

                ;; if (p < 0) p -= nudge else p += nudge   (rounding)
                local.get $p
                i64.const 0
                i64.lt_s
                (if
                  (then
                    local.get $p
                    local.get $nudge
                    i64.sub
                    local.set $p
                  )
                  (else
                    local.get $p
                    local.get $nudge
                    i64.add
                    local.set $p
                  )
                )

                ;; y = (i32)(p >> 31)
                local.get $p
                i64.const 31
                i64.shr_s
                i32.wrap_i64
                local.set $y

                ;; y += zy   (zp_y)
                local.get $y
                local.get $zy
                i32.add
                local.set $y

                ;; -------- activation --------
                ;; act == RELU (1): y = max(y, zy)
                local.get $act
                i32.const 1
                i32.eq
                (if
                  (then
                    local.get $y
                    local.get $zy
                    i32.lt_s
                    (if
                      (then
                        local.get $zy
                        local.set $y
                      )
                    )
                  )
                )

                ;; act == RELU6 (3): clamp y to [zy, q6_abs]
                local.get $act
                i32.const 3
                i32.eq
                (if
                  (then
                    ;; lo = zy
                    local.get $zy
                    local.set $lo

                    ;; hi = q6   (já é q6_abs = round(6/sy) + zy)
                    local.get $q6
                    local.set $hi

                    ;; hi = min(hi, 127)
                    local.get $hi
                    i32.const 127
                    i32.gt_s
                    (if
                      (then
                        i32.const 127
                        local.set $hi
                      )
                    )

                    ;; lo = max(lo, -128)
                    local.get $lo
                    i32.const -128
                    i32.lt_s
                    (if
                      (then
                        i32.const -128
                        local.set $lo
                      )
                    )

                    ;; if (y < lo) y = lo
                    local.get $y
                    local.get $lo
                    i32.lt_s
                    (if
                      (then
                        local.get $lo
                        local.set $y
                      )
                    )

                    ;; if (y > hi) y = hi
                    local.get $y
                    local.get $hi
                    i32.gt_s
                    (if
                      (then
                        local.get $hi
                        local.set $y
                      )
                    )
                  )
                )

                ;; -------- final clamp [-128..127] --------
                local.get $y
                i32.const 127
                i32.gt_s
                (if
                  (then
                    i32.const 127
                    local.set $y
                  )
                )

                local.get $y
                i32.const -128
                i32.lt_s
                (if
                  (then
                    i32.const -128
                    local.set $y
                  )
                )

                ;; store8(out_ptr[oc][outIdx]) = (i8)y
                local.get $baseOutOC
                local.get $outIdx
                i32.add
                local.get $y
                i32.store8


                ;; j_out++
                local.get $j_out
                i32.const 1
                i32.add
                local.set $j_out

                br $loop_j
              )
            )

            ;; i_out++
            local.get $i_out
            i32.const 1
            i32.add
            local.set $i_out

            br $loop_i
          )
        )

        ;; oc++
        local.get $oc
        i32.const 1
        i32.add
        local.set $oc

        br $loop_oc
      )
    )
  )

  (func $conv2d (export "conv2d") (param $layer_idx i32)
    (local $base     i32)

    (local $op_type   i32)
    (local $act       i32)
    (local $flags     i32)
    (local $in_ptr    i32)
    (local $out_ptr   i32)
    (local $in_h      i32)
    (local $in_w      i32)
    (local $cin       i32)
    (local $cout      i32)
    (local $kh        i32)
    (local $kw        i32)
    (local $stride_h  i32)
    (local $stride_w  i32)
    (local $dil_h     i32)
    (local $dil_w     i32)
    (local $pad_t     i32)
    (local $pad_b     i32)
    (local $pad_l     i32)
    (local $pad_r     i32)
    (local $wptr      i32)
    (local $bias_ptr  i32)
    (local $mul_ptr   i32)
    (local $q6_ptr    i32)
    (local $zx        i32)
    (local $zw        i32)
    (local $zy        i32)
    (local $out_h     i32)
    (local $out_w     i32)

    (local $bottom    i32)
    (local $right     i32)
    (local $plane_out i32)
    (local $w_per_oc  i32)

    (local $oc        i32)
    (local $i_out     i32)
    (local $j_out     i32)
    (local $ki        i32)
    (local $kj        i32)

    (local $baseOutOC i32)
    (local $baseWoc   i32)

    (local $b         i32)
    (local $m         i32)
    (local $q6        i32)

    (local $i         i32)
    (local $j         i32)

    (local $acc       i32)

    (local $row       i32)
    (local $col       i32)

    (local $row_img   i32)
    (local $row_base  i32)
    (local $col_img   i32)
    (local $idx       i32)

    (local $pos       i32)
    (local $baseW_kikj i32)

    (local $outIdx    i32)

    (local $tmp i32)
    (local $w0  i32)

    (local $y   i32)
    (local $hi  i32)
    (local $lo  i32)
    (local $p   i64)
    (local $nudge i64)

    (local $c i32)

    ;; base = PARAMS_BASE + layer_idx * LP_SIZE
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    ;; -------------------------
    ;; helpers inline:
    ;; load_i32(field_idx) = i32.load(base + field_idx*4)
    ;; -------------------------

    ;; op_type (0)
    local.get $base
    i32.const 0
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $op_type

    ;; act (1)
    local.get $base
    i32.const 1
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $act

    ;; flags (2)
    local.get $base
    i32.const 2
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $flags

    ;; in_ptr (3)
    local.get $base
    i32.const 3
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr

    ;; out_ptr (4)
    local.get $base
    i32.const 4
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_ptr

    ;; in_h (5)
    local.get $base
    i32.const 5
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_h

    ;; in_w (6)
    local.get $base
    i32.const 6
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_w

    ;; cin (7)
    local.get $base
    i32.const 7
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cin

    ;; cout (8)
    local.get $base
    i32.const 8
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cout

    ;; kh (9)
    local.get $base
    i32.const 9
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $kh

    ;; kw (10)
    local.get $base
    i32.const 10
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $kw

    ;; stride_h (11)
    local.get $base
    i32.const 11
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $stride_h

    ;; stride_w (12)
    local.get $base
    i32.const 12
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $stride_w

    ;; dil_h (13)
    local.get $base
    i32.const 13
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $dil_h

    ;; dil_w (14)
    local.get $base
    i32.const 14
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $dil_w

    ;; pad_t (15)
    local.get $base
    i32.const 15
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_t

    ;; pad_b (16)
    local.get $base
    i32.const 16
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_b

    ;; pad_l (17)
    local.get $base
    i32.const 17
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_l

    ;; pad_r (18)
    local.get $base
    i32.const 18
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $pad_r

    ;; wptr (19)
    local.get $base
    i32.const 19
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $wptr

    ;; bias_ptr (20)
    local.get $base
    i32.const 20
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $bias_ptr

    ;; mul_ptr (21)
    local.get $base
    i32.const 21
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $mul_ptr

    ;; q6_ptr (22)
    local.get $base
    i32.const 22
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $q6_ptr

    ;; zx (23)
    local.get $base
    i32.const 23
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zx

    ;; zw (24)
    local.get $base
    i32.const 24
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zw

    ;; zy (25)
    local.get $base
    i32.const 25
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zy

    ;; out_h (26)
    local.get $base
    i32.const 26
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_h

    ;; out_w (27)
    local.get $base
    i32.const 27
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_w

    ;; =========================


    ;; bottom = pad_t + in_h;
    local.get $pad_t
    local.get $in_h
    i32.add
    local.set $bottom

    ;; right = pad_l + in_w;
    local.get $pad_l
    local.get $in_w
    i32.add
    local.set $right

    ;; plane_out = out_h * out_w;
    local.get $out_h
    local.get $out_w
    i32.mul
    local.set $plane_out

    ;; // bytes por oc: kh*kw*cin = 9*3 = 27
    local.get $kh
    local.get $kw
    local.get $cin
    i32.mul
    i32.mul
    local.set $w_per_oc

    ;; =========================
    ;; loops: oc / i_out / j_out / ki / kj
    ;; com dilatação: row = i + ki*dil_h, col = j + kj*dil_w
    ;; =========================

    ;; oc = 0
    i32.const 0
    local.set $oc

    (block $exit_oc
      (loop $loop_oc

        ;; if (oc >= cout) break;
        local.get $oc
        local.get $cout
        i32.ge_s
        br_if $exit_oc

        ;; baseOutOC = out_ptr + oc * plane_out
        local.get $out_ptr
        local.get $oc
        local.get $plane_out
        i32.mul
        i32.add
        local.set $baseOutOC

        ;; b = load<i32>(bias_ptr + oc*4)
        local.get $bias_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $b

        ;; m = load<i32>(mul_ptr + oc*4)
        local.get $mul_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $m

        ;; q6 = (q6_ptr != 0) ? load<i32>(q6_ptr + oc*4) : 0
        local.get $q6_ptr
        i32.eqz
        (if (result i32)
          (then
            i32.const 0
          )
          (else
            local.get $q6_ptr
            local.get $oc
            i32.const 2
            i32.shl
            i32.add
            i32.load align=4
          )
        )
        local.set $q6

        ;; baseWoc = wptr + oc * w_per_oc
        local.get $wptr
        local.get $oc
        local.get $w_per_oc
        i32.mul
        i32.add
        local.set $baseWoc

        ;; i_out = 0
        i32.const 0
        local.set $i_out

        (block $exit_i
          (loop $loop_i

            ;; if (i_out >= out_h) break;
            local.get $i_out
            local.get $out_h
            i32.ge_s
            br_if $exit_i

            ;; i = i_out * stride_h
            local.get $i_out
            local.get $stride_h
            i32.mul
            local.set $i

            ;; j_out = 0
            i32.const 0
            local.set $j_out

            (block $exit_j
              (loop $loop_j

                ;; if (j_out >= out_w) break;
                local.get $j_out
                local.get $out_w
                i32.ge_s
                br_if $exit_j

                ;; j = j_out * stride_w
                local.get $j_out
                local.get $stride_w
                i32.mul
                local.set $j

                ;; acc = b
                local.get $b
                local.set $acc

                ;; ki = 0
                i32.const 0
                local.set $ki

                (block $exit_ki
                  (loop $loop_ki

                    ;; if (ki >= kh) break;
                    local.get $ki
                    local.get $kh
                    i32.ge_s
                    br_if $exit_ki

                    ;; ---- alvo do "continue" do ki ----
                    (block $inc_ki

                      ;; row = i + ki*dil_h   (DILATAÇÃO AQUI)
                      local.get $i
                      local.get $ki
                      local.get $dil_h
                      i32.mul
                      i32.add
                      local.set $row

                      ;; if (row < pad_t) continue;
                      local.get $row
                      local.get $pad_t
                      i32.lt_s
                      br_if $inc_ki

                      ;; if (row >= bottom) break;
                      local.get $row
                      local.get $bottom
                      i32.ge_s
                      br_if $exit_ki

                      ;; row_img  = row - pad_t
                      local.get $row
                      local.get $pad_t
                      i32.sub
                      local.set $row_img

                      ;; if (row_img < 0 || row_img >= in_h) continue;
                      local.get $row_img
                      i32.const 0
                      i32.lt_s
                      br_if $inc_ki

                      local.get $row_img
                      local.get $in_h
                      i32.ge_s
                      br_if $inc_ki

                      ;; row_base = row_img * in_w
                      local.get $row_img
                      local.get $in_w
                      i32.mul
                      local.set $row_base

                      ;; kj = 0
                      i32.const 0
                      local.set $kj

                      (block $exit_kj
                        (loop $loop_kj

                          ;; if (kj >= kw) break;
                          local.get $kj
                          local.get $kw
                          i32.ge_s
                          br_if $exit_kj

                          ;; ---- alvo do "continue" do kj ----
                          (block $inc_kj

                            ;; col = j + kj*dil_w   (DILATAÇÃO AQUI)
                            local.get $j
                            local.get $kj
                            local.get $dil_w
                            i32.mul
                            i32.add
                            local.set $col

                            ;; if (col < pad_l) continue;
                            local.get $col
                            local.get $pad_l
                            i32.lt_s
                            br_if $inc_kj

                            ;; if (col >= right) break;
                            local.get $col
                            local.get $right
                            i32.ge_s
                            br_if $exit_kj

                            ;; col_img = col - pad_l
                            local.get $col
                            local.get $pad_l
                            i32.sub
                            local.set $col_img

                            ;; if (col_img < 0 || col_img >= in_w) continue;
                            local.get $col_img
                            i32.const 0
                            i32.lt_s
                            br_if $inc_kj

                            local.get $col_img
                            local.get $in_w
                            i32.ge_s
                            br_if $inc_kj

                            ;; for c in [0..cin)
                            i32.const 0
                            local.set $c

                            (block $exit_c
                              (loop $loop_c

                                ;; if (c >= cin) break
                                local.get $c
                                local.get $cin
                                i32.ge_s
                                br_if $exit_c

                                ;; input_idx = (row_base + col_img) * cin + c
                                local.get $row_base
                                local.get $col_img
                                i32.add
                                local.get $cin
                                i32.mul
                                local.get $c
                                i32.add
                                local.set $idx

                                ;; tmp = load<i8>(in_ptr + idx)
                                local.get $in_ptr
                                local.get $idx
                                i32.add
                                i32.load8_s
                                local.set $tmp

                                ;; pos = ((ki*kw + kj) * cin) + c
                                local.get $ki
                                local.get $kw
                                i32.mul
                                local.get $kj
                                i32.add
                                local.get $cin
                                i32.mul
                                local.get $c
                                i32.add
                                local.set $pos

                                ;; w0 = load<i8>(baseWoc + pos)
                                local.get $baseWoc
                                local.get $pos
                                i32.add
                                i32.load8_s
                                local.set $w0

                                ;; acc += (tmp - zx) * (w0 - zw)
                                local.get $acc
                                local.get $tmp
                                local.get $zx
                                i32.sub
                                local.get $w0
                                local.get $zw
                                i32.sub
                                i32.mul
                                i32.add
                                local.set $acc

                                ;; c++
                                local.get $c
                                i32.const 1
                                i32.add
                                local.set $c

                                br $loop_c
                              )
                            )

                            ;; kj++
                            local.get $kj
                            i32.const 1
                            i32.add
                            local.set $kj

                            br $loop_kj
                          )
                        )
                      )
                    )

                    ;; ki++
                    local.get $ki
                    i32.const 1
                    i32.add
                    local.set $ki
                    br $loop_ki
                  )
                )

                ;; outIdx = i_out*out_w + j_out
                local.get $i_out
                local.get $out_w
                i32.mul
                local.get $j_out
                i32.add
                local.set $outIdx

                ;; =========================
                ;; REQUANT + zp_y + activation + clamp + store8
                ;; =========================

                ;; p = (i64)acc * (i64)m
                local.get $acc
                i64.extend_i32_s
                local.get $m
                i64.extend_i32_s
                i64.mul
                local.set $p

                ;; RSHIFT = 31
                ;; nudge = 1 << (31-1) = 1<<30
                i64.const 1
                i64.const 30
                i64.shl
                local.set $nudge

                ;; if (p < 0) p -= nudge else p += nudge   (rounding)
                local.get $p
                i64.const 0
                i64.lt_s
                (if
                  (then
                    local.get $p
                    local.get $nudge
                    i64.sub
                    local.set $p
                  )
                  (else
                    local.get $p
                    local.get $nudge
                    i64.add
                    local.set $p
                  )
                )

                ;; y = (i32)(p >> 31)
                local.get $p
                i64.const 31
                i64.shr_s
                i32.wrap_i64
                local.set $y

                ;; y += zy   (zp_y)
                local.get $y
                local.get $zy
                i32.add
                local.set $y

                ;; -------- activation --------
                ;; act == RELU (1): y = max(y, zy)
                local.get $act
                i32.const 1
                i32.eq
                (if
                  (then
                    local.get $y
                    local.get $zy
                    i32.lt_s
                    (if
                      (then
                        local.get $zy
                        local.set $y
                      )
                    )
                  )
                )
                
                ;; act == RELU6 (3): clamp y to [zy, q6_abs]
                local.get $act
                i32.const 3
                i32.eq
                (if
                  (then
                    ;; lo = zy
                    local.get $zy
                    local.set $lo
                
                    ;; hi = q6   (já é q6_abs = round(6/sy) + zy)
                    local.get $q6
                    local.set $hi
                
                    ;; hi = min(hi, 127)
                    local.get $hi
                    i32.const 127
                    i32.gt_s
                    (if
                      (then
                        i32.const 127
                        local.set $hi
                      )
                    )
                
                    ;; lo = max(lo, -128)
                    local.get $lo
                    i32.const -128
                    i32.lt_s
                    (if
                      (then
                        i32.const -128
                        local.set $lo
                      )
                    )
                
                    ;; if (y < lo) y = lo
                    local.get $y
                    local.get $lo
                    i32.lt_s
                    (if
                      (then
                        local.get $lo
                        local.set $y
                      )
                    )
                
                    ;; if (y > hi) y = hi
                    local.get $y
                    local.get $hi
                    i32.gt_s
                    (if
                      (then
                        local.get $hi
                        local.set $y
                      )
                    )
                  )
                )

                ;; -------- final clamp [-128..127] --------
                local.get $y
                i32.const 127
                i32.gt_s
                (if
                  (then
                    i32.const 127
                    local.set $y
                  )
                )

                local.get $y
                i32.const -128
                i32.lt_s
                (if
                  (then
                    i32.const -128
                    local.set $y
                  )
                )

                ;; store8(out_ptr[oc][outIdx]) = (i8)y
                local.get $baseOutOC
                local.get $outIdx
                i32.add
                local.get $y
                i32.store8


                ;; j_out++
                local.get $j_out
                i32.const 1
                i32.add
                local.set $j_out

                br $loop_j
              )
            )

            ;; i_out++
            local.get $i_out
            i32.const 1
            i32.add
            local.set $i_out

            br $loop_i
          )
        )

        ;; oc++
        local.get $oc
        i32.const 1
        i32.add
        local.set $oc

        br $loop_oc
      )
    )
  )

  ;; ============================================================
  ;; ADD Layer - CORRIGIDO com Quantização Q31 Completa
  ;; Campos usados:
  ;;   kh = mul0 (Q31), kw = shift0
  ;;   stride_h = mul1 (Q31), stride_w = shift1
  ;;   dil_h = out_mul (Q31), dil_w = out_shift
  ;;   pad_t = input_ptr[0], pad_b = input_ptr[1]
  ;;   pad_l = zA, pad_r = zB
  ;;   zx = zA, zw = zB, zy = zY
  ;; ============================================================
  (func $add (export "add") (param $layer_idx i32)
    (local $base     i32)
    (local $in_ptr_0 i32)  ;; de pad_t
    (local $in_ptr_1 i32)  ;; de pad_b
    (local $out_ptr  i32)
    (local $in_h     i32)
    (local $in_w     i32)
    (local $cin      i32)
    
    ;; Parâmetros de quantização
    (local $mul0     i32)  ;; de kh
    (local $shift0   i32)  ;; de kw
    (local $mul1     i32)  ;; de stride_h
    (local $shift1   i32)  ;; de stride_w
    (local $out_mul  i32)  ;; de dil_h
    (local $out_shift i32) ;; de dil_w
    (local $zA       i32)  ;; de pad_l
    (local $zB       i32)  ;; de pad_r
    (local $zY       i32)  ;; de zy
    
    (local $total    i32)
    (local $idx      i32)
    (local $val0     i32)
    (local $val1     i32)
    (local $acc0     i32)
    (local $acc1     i32)
    (local $sum      i32)
    (local $result   i32)

    ;; Carregar base
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    ;; pad_t (15) = in_ptr_0
    local.get $base
    i32.const 15
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr_0

    ;; pad_b (16) = in_ptr_1
    local.get $base
    i32.const 16
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr_1

    ;; out_ptr (4)
    local.get $base
    i32.const 4
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_ptr

    ;; in_h (5)
    local.get $base
    i32.const 5
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_h

    ;; in_w (6)
    local.get $base
    i32.const 6
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_w

    ;; cin (7)
    local.get $base
    i32.const 7
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cin

    ;; kh (9) = mul0
    local.get $base
    i32.const 9
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $mul0

    ;; kw (10) = shift0
    local.get $base
    i32.const 10
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $shift0

    ;; stride_h (11) = mul1
    local.get $base
    i32.const 11
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $mul1

    ;; stride_w (12) = shift1
    local.get $base
    i32.const 12
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $shift1

    ;; dil_h (13) = out_mul
    local.get $base
    i32.const 13
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_mul

    ;; dil_w (14) = out_shift
    local.get $base
    i32.const 14
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_shift

    ;; pad_l (17) = zA
    local.get $base
    i32.const 17
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zA

    ;; pad_r (18) = zB
    local.get $base
    i32.const 18
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zB

    ;; zy (25) = zY
    local.get $base
    i32.const 25
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zY

    ;; total = in_h * in_w * cin
    local.get $in_h
    local.get $in_w
    i32.mul
    local.get $cin
    i32.mul
    local.set $total

    ;; Loop através de todos os elementos
    i32.const 0
    local.set $idx

    (block $exit_loop
      (loop $loop
        ;; if (idx >= total) break
        local.get $idx
        local.get $total
        i32.ge_s
        br_if $exit_loop

        ;; val0 = load<i8>(in_ptr_0 + idx) - zA
        local.get $in_ptr_0
        local.get $idx
        i32.add
        i32.load8_s
        local.get $zA
        i32.sub
        local.set $val0

        ;; val1 = load<i8>(in_ptr_1 + idx) - zB
        local.get $in_ptr_1
        local.get $idx
        i32.add
        i32.load8_s
        local.get $zB
        i32.sub
        local.set $val1

        ;; acc0 = multiply_by_quantized_multiplier(val0, mul0, shift0)
        local.get $val0
        local.get $mul0
        local.get $shift0
        call $multiply_by_quantized_multiplier
        local.set $acc0

        ;; acc1 = multiply_by_quantized_multiplier(val1, mul1, shift1)
        local.get $val1
        local.get $mul1
        local.get $shift1
        call $multiply_by_quantized_multiplier
        local.set $acc1

        ;; sum = acc0 + acc1
        local.get $acc0
        local.get $acc1
        i32.add
        local.set $sum

        ;; result = multiply_by_quantized_multiplier(sum, out_mul, out_shift)
        local.get $sum
        local.get $out_mul
        local.get $out_shift
        call $multiply_by_quantized_multiplier
        local.set $result

        ;; result += zY
        local.get $result
        local.get $zY
        i32.add
        local.set $result

        ;; Clamp [-128, 127]
        local.get $result
        i32.const 127
        i32.gt_s
        (if
          (then
            i32.const 127
            local.set $result
          )
        )

        local.get $result
        i32.const -128
        i32.lt_s
        (if
          (then
            i32.const -128
            local.set $result
          )
        )

        ;; store<i8>(out_ptr + idx) = result
        local.get $out_ptr
        local.get $idx
        i32.add
        local.get $result
        i32.store8

        ;; idx++
        local.get $idx
        i32.const 1
        i32.add
        local.set $idx

        br $loop
      )
    )
  )

  ;; ============================================================
  ;; MEAN Layer - CORRIGIDO com Quantização
  ;; Campos usados:
  ;;   kh = mul (Q31), kw = shift
  ;;   stride_h = spatial_size (in_h * in_w)
  ;;   pad_t = input_ptr
  ;;   zx, zy para zero points
  ;; ============================================================
  (func $mean (export "mean") (param $layer_idx i32)
    (local $base     i32)
    (local $in_ptr   i32)
    (local $out_ptr  i32)
    (local $in_h     i32)
    (local $in_w     i32)
    (local $cin      i32)
    (local $zX       i32)
    (local $zY       i32)
    (local $mul      i32)
    (local $shift    i32)
    (local $spatial_size i32)
    
    (local $c        i32)
    (local $idx      i32)
    (local $sum      i32)
    (local $val      i32)
    (local $mean_val i32)
    (local $result   i32)

    ;; Carregar base
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    ;; pad_t (15) = in_ptr para MEAN
    local.get $base
    i32.const 15
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr

    ;; out_ptr (4)
    local.get $base
    i32.const 4
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_ptr

    ;; in_h (5)
    local.get $base
    i32.const 5
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_h

    ;; in_w (6)
    local.get $base
    i32.const 6
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_w

    ;; cin (7)
    local.get $base
    i32.const 7
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cin

    ;; kh (9) = mul
    local.get $base
    i32.const 9
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $mul

    ;; kw (10) = shift
    local.get $base
    i32.const 10
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $shift

    ;; stride_h (11) = spatial_size
    local.get $base
    i32.const 11
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $spatial_size

    ;; zx (23)
    local.get $base
    i32.const 23
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zX

    ;; zy (25)
    local.get $base
    i32.const 25
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zY

    ;; Para cada canal
    i32.const 0
    local.set $c

    (block $exit_c
      (loop $loop_c
        ;; if (c >= cin) break
        local.get $c
        local.get $cin
        i32.ge_s
        br_if $exit_c

        ;; sum = 0
        i32.const 0
        local.set $sum

        ;; Somar todos os valores espaciais para este canal
        i32.const 0
        local.set $idx

        (block $exit_spatial
          (loop $loop_spatial
            ;; if (idx >= spatial_size) break
            local.get $idx
            local.get $spatial_size
            i32.ge_s
            br_if $exit_spatial

            ;; val = load<i8>(in_ptr + (idx * cin + c)) - zX
            local.get $in_ptr
            local.get $idx
            local.get $cin
            i32.mul
            local.get $c
            i32.add
            i32.add
            i32.load8_s
            local.get $zX
            i32.sub
            local.set $val

            ;; sum += val
            local.get $sum
            local.get $val
            i32.add
            local.set $sum

            ;; idx++
            local.get $idx
            i32.const 1
            i32.add
            local.set $idx

            br $loop_spatial
          )
        )

        ;; mean_val = sum / spatial_size (sem quantização primeiro)
        local.get $sum
        local.get $spatial_size
        i32.div_s
        local.set $mean_val

        ;; Aplicar quantização: multiply_by_quantized_multiplier(mean_val, mul, shift)
        local.get $mean_val
        local.get $mul
        local.get $shift
        call $multiply_by_quantized_multiplier
        local.set $result

        ;; result += zY
        local.get $result
        local.get $zY
        i32.add
        local.set $result

        ;; Clamp [-128, 127]
        local.get $result
        i32.const 127
        i32.gt_s
        (if
          (then
            i32.const 127
            local.set $result
          )
        )

        local.get $result
        i32.const -128
        i32.lt_s
        (if
          (then
            i32.const -128
            local.set $result
          )
        )

        ;; store<i8>(out_ptr + c) = result
        local.get $out_ptr
        local.get $c
        i32.add
        local.get $result
        i32.store8

        ;; c++
        local.get $c
        i32.const 1
        i32.add
        local.set $c

        br $loop_c
      )
    )
  )

  ;; ============================================================
  ;; FULLY_CONNECTED Layer
  ;; Multiplicação matriz-vetor: y = Wx + b
  ;; ============================================================
  (func $fully_connected (export "fully_connected") (param $layer_idx i32)
    (local $base     i32)
    (local $in_ptr   i32)
    (local $out_ptr  i32)
    (local $cin      i32)
    (local $cout     i32)
    (local $wptr     i32)
    (local $bias_ptr i32)
    (local $mul_ptr  i32)
    (local $zx       i32)
    (local $zw       i32)
    (local $zy       i32)
    
    (local $oc       i32)
    (local $ic       i32)
    (local $acc      i32)
    (local $b        i32)
    (local $m        i32)
    (local $input_val i32)
    (local $weight_val i32)
    (local $y        i32)
    (local $p        i64)
    (local $nudge    i64)

    ;; Carregar base
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    ;; in_ptr (3)
    local.get $base
    i32.const 3
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr

    ;; out_ptr (4)
    local.get $base
    i32.const 4
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_ptr

    ;; cin (7)
    local.get $base
    i32.const 7
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cin

    ;; cout (8)
    local.get $base
    i32.const 8
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cout

    ;; wptr (19)
    local.get $base
    i32.const 19
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $wptr

    ;; bias_ptr (20)
    local.get $base
    i32.const 20
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $bias_ptr

    ;; mul_ptr (21)
    local.get $base
    i32.const 21
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $mul_ptr

    ;; zx (23)
    local.get $base
    i32.const 23
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zx

    ;; zw (24)
    local.get $base
    i32.const 24
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zw

    ;; zy (25)
    local.get $base
    i32.const 25
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zy

    ;; Para cada neurônio de saída
    i32.const 0
    local.set $oc

    (block $exit_oc
      (loop $loop_oc
        ;; if (oc >= cout) break
        local.get $oc
        local.get $cout
        i32.ge_s
        br_if $exit_oc

        ;; b = load<i32>(bias_ptr + oc*4)
        local.get $bias_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $b

        ;; m = load<i32>(mul_ptr + oc*4)
        local.get $mul_ptr
        local.get $oc
        i32.const 2
        i32.shl
        i32.add
        i32.load align=4
        local.set $m

        ;; acc = b
        local.get $b
        local.set $acc

        ;; Para cada entrada
        i32.const 0
        local.set $ic

        (block $exit_ic
          (loop $loop_ic
            ;; if (ic >= cin) break
            local.get $ic
            local.get $cin
            i32.ge_s
            br_if $exit_ic

            ;; input_val = load<i8>(in_ptr + ic)
            local.get $in_ptr
            local.get $ic
            i32.add
            i32.load8_s
            local.set $input_val

            ;; weight_val = load<i8>(wptr + oc*cin + ic)
            local.get $wptr
            local.get $oc
            local.get $cin
            i32.mul
            local.get $ic
            i32.add
            i32.add
            i32.load8_s
            local.set $weight_val

            ;; acc += (input_val - zx) * (weight_val - zw)
            local.get $acc
            local.get $input_val
            local.get $zx
            i32.sub
            local.get $weight_val
            local.get $zw
            i32.sub
            i32.mul
            i32.add
            local.set $acc

            ;; ic++
            local.get $ic
            i32.const 1
            i32.add
            local.set $ic

            br $loop_ic
          )
        )

        ;; Requantização
        ;; p = (i64)acc * (i64)m
        local.get $acc
        i64.extend_i32_s
        local.get $m
        i64.extend_i32_s
        i64.mul
        local.set $p

        ;; nudge = 1 << 30
        i64.const 1
        i64.const 30
        i64.shl
        local.set $nudge

        ;; Rounding
        local.get $p
        i64.const 0
        i64.lt_s
        (if
          (then
            local.get $p
            local.get $nudge
            i64.sub
            local.set $p
          )
          (else
            local.get $p
            local.get $nudge
            i64.add
            local.set $p
          )
        )

        ;; y = (i32)(p >> 31)
        local.get $p
        i64.const 31
        i64.shr_s
        i32.wrap_i64
        local.set $y

        ;; y += zy
        local.get $y
        local.get $zy
        i32.add
        local.set $y

        ;; Clamp [-128, 127]
        local.get $y
        i32.const 127
        i32.gt_s
        (if
          (then
            i32.const 127
            local.set $y
          )
        )

        local.get $y
        i32.const -128
        i32.lt_s
        (if
          (then
            i32.const -128
            local.set $y
          )
        )

        ;; store<i8>(out_ptr + oc) = y
        local.get $out_ptr
        local.get $oc
        i32.add
        local.get $y
        i32.store8

        ;; oc++
        local.get $oc
        i32.const 1
        i32.add
        local.set $oc

        br $loop_oc
      )
    )
  )

  ;; ============================================================
  ;; SOFTMAX Layer - CORRIGIDO com parâmetros beta
  ;; Campos usados:
  ;;   kh = input_beta_mul (Q31)
  ;;   kw = input_beta_left_shift
  ;;   stride_h = diff_min
  ;;   stride_w = integer_bits
  ;;   pad_t = input_ptr
  ;;   zx = zX, zy = zY
  ;; ============================================================
  (func $softmax (export "softmax") (param $layer_idx i32)
    (local $base     i32)
    (local $in_ptr   i32)
    (local $out_ptr  i32)
    (local $cin      i32)
    (local $zX       i32)
    (local $zY       i32)
    (local $input_beta_mul i32)
    (local $input_beta_shift i32)
    (local $diff_min i32)
    (local $integer_bits i32)
    
    (local $i        i32)
    (local $max_val  i32)
    (local $val      i32)
    (local $scaled_val i32)
    (local $sum      i32)
    (local $exp_val  i32)
    (local $out_val  i32)

    ;; Carregar base
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    ;; pad_t (15) = in_ptr para SOFTMAX
    local.get $base
    i32.const 15
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $in_ptr

    ;; out_ptr (4)
    local.get $base
    i32.const 4
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $out_ptr

    ;; cin (7)
    local.get $base
    i32.const 7
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $cin

    ;; kh (9) = input_beta_mul
    local.get $base
    i32.const 9
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $input_beta_mul

    ;; kw (10) = input_beta_shift
    local.get $base
    i32.const 10
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $input_beta_shift

    ;; stride_h (11) = diff_min
    local.get $base
    i32.const 11
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $diff_min

    ;; stride_w (12) = integer_bits
    local.get $base
    i32.const 12
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $integer_bits

    ;; zx (23)
    local.get $base
    i32.const 23
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zX

    ;; zy (25)
    local.get $base
    i32.const 25
    i32.const 2
    i32.shl
    i32.add
    i32.load align=4
    local.set $zY

    ;; Encontrar valor máximo (para estabilidade numérica)
    i32.const -128
    local.set $max_val

    i32.const 0
    local.set $i

    (block $exit_max
      (loop $loop_max
        local.get $i
        local.get $cin
        i32.ge_s
        br_if $exit_max

        local.get $in_ptr
        local.get $i
        i32.add
        i32.load8_s
        local.set $val

        local.get $val
        local.get $max_val
        i32.gt_s
        (if
          (then
            local.get $val
            local.set $max_val
          )
        )

        local.get $i
        i32.const 1
        i32.add
        local.set $i

        br $loop_max
      )
    )

    ;; Softmax quantizado simplificado
    ;; Para int8, usamos aproximação via escala
    i32.const 0
    local.set $i

    (block $exit_copy
      (loop $loop_copy
        local.get $i
        local.get $cin
        i32.ge_s
        br_if $exit_copy

        ;; val = load - zX
        local.get $in_ptr
        local.get $i
        i32.add
        i32.load8_s
        local.get $zX
        i32.sub
        local.set $val

        ;; Aplicar beta scaling: multiply_by_quantized_multiplier
        local.get $val
        local.get $input_beta_mul
        local.get $input_beta_shift
        call $multiply_by_quantized_multiplier
        local.set $scaled_val

        ;; Clamp com diff_min
        local.get $scaled_val
        local.get $diff_min
        i32.lt_s
        (if
          (then
            local.get $diff_min
            local.set $scaled_val
          )
        )

        ;; Adicionar zY
        local.get $scaled_val
        local.get $zY
        i32.add
        local.set $out_val

        ;; Clamp final
        local.get $out_val
        i32.const 127
        i32.gt_s
        (if
          (then
            i32.const 127
            local.set $out_val
          )
        )

        local.get $out_val
        i32.const -128
        i32.lt_s
        (if
          (then
            i32.const -128
            local.set $out_val
          )
        )

        ;; store
        local.get $out_ptr
        local.get $i
        i32.add
        local.get $out_val
        i32.store8

        local.get $i
        i32.const 1
        i32.add
        local.set $i

        br $loop_copy
      )
    )
  )

  ;; ============================================================
  ;; Dispatcher - Executa a camada correta baseado no layer_idx
  ;; ============================================================
  (func $run_layer (export "run_layer") (param $layer_idx i32)
    (local $base i32)
    (local $op_type i32)

    ;; Carregar op_type
    local.get $layer_idx
    call $layerparam_base
    local.set $base

    local.get $base
    i32.load align=4
    local.set $op_type

    ;; Switch baseado em op_type
    ;; 1 = CONV, 2 = DW, 3 = FC, 4 = ADD, 5 = MEAN, 6 = SOFTMAX

    local.get $op_type
    i32.const 1
    i32.eq
    (if
      (then
        ;; Primeira camada usa conv2d_layer0 (RGB565)
        local.get $layer_idx
        i32.const 0
        i32.eq
        (if
          (then
            local.get $layer_idx
            call $conv2d_layer0
            return
          )
        )
        ;; Outras camadas CONV usam conv2d genérico
        local.get $layer_idx
        call $conv2d
        return
      )
    )

    local.get $op_type
    i32.const 2
    i32.eq
    (if
      (then
        local.get $layer_idx
        call $depthwise_conv2d
        return
      )
    )

    local.get $op_type
    i32.const 3
    i32.eq
    (if
      (then
        local.get $layer_idx
        call $fully_connected
        return
      )
    )

    local.get $op_type
    i32.const 4
    i32.eq
    (if
      (then
        local.get $layer_idx
        call $add
        return
      )
    )

    local.get $op_type
    i32.const 5
    i32.eq
    (if
      (then
        local.get $layer_idx
        call $mean
        return
      )
    )

    local.get $op_type
    i32.const 6
    i32.eq
    (if
      (then
        local.get $layer_idx
        call $softmax
        return
      )
    )
  )

  ;; ============================================================
  ;; Executa toda a rede MobileNetV2
  ;; ============================================================
  (func $run_mobilenetv2 (export "run_mobilenetv2") (result i32)
    ;; Executar todas as 65 camadas em sequência
    i32.const 0
    call $run_layer
    
    i32.const 1
    call $run_layer
    
    i32.const 2
    call $run_layer
    
    i32.const 3
    call $run_layer
    
    i32.const 4
    call $run_layer
    
    i32.const 5
    call $run_layer
    
    i32.const 6
    call $run_layer
    
    i32.const 7
    call $run_layer
    
    i32.const 8
    call $run_layer
    
    i32.const 9
    call $run_layer
    
    i32.const 10
    call $run_layer
    
    i32.const 11
    call $run_layer
    
    i32.const 12
    call $run_layer
    
    i32.const 13
    call $run_layer
    
    i32.const 14
    call $run_layer
    
    i32.const 15
    call $run_layer
    
    i32.const 16
    call $run_layer
    
    i32.const 17
    call $run_layer
    
    i32.const 18
    call $run_layer
    
    i32.const 19
    call $run_layer
    
    i32.const 20
    call $run_layer
    
    i32.const 21
    call $run_layer
    
    i32.const 22
    call $run_layer
    
    i32.const 23
    call $run_layer
    
    i32.const 24
    call $run_layer
    
    i32.const 25
    call $run_layer
    
    i32.const 26
    call $run_layer
    
    i32.const 27
    call $run_layer
    
    i32.const 28
    call $run_layer
    
    i32.const 29
    call $run_layer
    
    i32.const 30
    call $run_layer
    
    i32.const 31
    call $run_layer
    
    i32.const 32
    call $run_layer
    
    i32.const 33
    call $run_layer
    
    i32.const 34
    call $run_layer
    
    i32.const 35
    call $run_layer
    
    i32.const 36
    call $run_layer
    
    i32.const 37
    call $run_layer
    
    i32.const 38
    call $run_layer
    
    i32.const 39
    call $run_layer
    
    i32.const 40
    call $run_layer
    
    i32.const 41
    call $run_layer
    
    i32.const 42
    call $run_layer
    
    i32.const 43
    call $run_layer
    
    i32.const 44
    call $run_layer
    
    i32.const 45
    call $run_layer
    
    i32.const 46
    call $run_layer
    
    i32.const 47
    call $run_layer
    
    i32.const 48
    call $run_layer
    
    i32.const 49
    call $run_layer
    
    i32.const 50
    call $run_layer
    
    i32.const 51
    call $run_layer
    
    i32.const 52
    call $run_layer
    
    i32.const 53
    call $run_layer
    
    i32.const 54
    call $run_layer
    
    i32.const 55
    call $run_layer
    
    i32.const 56
    call $run_layer
    
    i32.const 57
    call $run_layer
    
    i32.const 58
    call $run_layer
    
    i32.const 59
    call $run_layer
    
    i32.const 60
    call $run_layer
    
    i32.const 61
    call $run_layer
    
    i32.const 62
    call $run_layer
    
    i32.const 63
    call $run_layer
    
    i32.const 64
    call $run_layer

    ;; Retorna 0 indicando sucesso
    i32.const 0
  )