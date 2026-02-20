import { readByte, readInt32, debugAddress, readInt32Signed, readRaw4, readInt8 } from "./memory_reader.js";

//console.log(readByte(2049));       // peso int8
//console.log(readInt32(1664096));   // bias int32
//console.log(readInt32Signed(1664100));   // bias int32
//console.log(readRaw4(1664100));   // -30175 => bias int32

//debugAddress(2049);

// camada 64 FC
// console.log("op_type = ", readInt32Signed(1800160));
// console.log("act = ", readInt32Signed(1800164));
// console.log("flags = ", readInt32Signed(1800168));
// console.log("in_ptr = " , readInt32Signed(1800172));
// console.log("out_ptr = ", readInt32Signed(1800176));
// console.log("in_h = ", readInt32Signed(1800180));
// console.log("in_w = ", readInt32Signed(1800184));
// console.log("cin = ", readInt32Signed(1800188));
// console.log("cout = ", readInt32Signed(1800192));
// console.log("kh = ", readInt32Signed(1800196));
// console.log("kw = ", readInt32Signed(1800200));
// console.log("stride_h = ", readInt32Signed(1800204));
// console.log("stride_w = ", readInt32Signed(1800208));
// console.log("dil_h = ", readInt32Signed(1800212));
// console.log("dil_w = ", readInt32Signed(1800216));
// console.log("pad_t = ", readInt32Signed(1800220));
// console.log("pad_b = ", readInt32Signed(1800224));
// console.log("pad_l = ", readInt32Signed(1800228));
// console.log("pad_r = ", readInt32Signed(1800232));
// console.log("wptr = ", readInt32Signed(1800236));
// console.log("bias_ptr = ", readInt32Signed(1800240));
// console.log("mul_ptr = ", readInt32Signed(1800244));
// console.log("shift_ptr = ", readInt32Signed(1800248));
// console.log("q6_ptr = ", readInt32Signed(1800252));
// console.log("zx = ", readInt32Signed(1800256));
// console.log("zw = ", readInt32Signed(1800260));
// console.log("zy = ", readInt32Signed(1800264));
// console.log("out_h = ", readInt32Signed(1800268));
// console.log("out_w = ", readInt32Signed(1800272));
// console.log("bias_ptr[1692256] = ", readInt32Signed(1692256)); // bias int32
// console.log("bias_ptr[1692260] = ", readInt32Signed(1692260)); // bias int32
// console.log("wptr[384096] = ", readInt8(384096)); // weight int8
// console.log("wptr[384097] = ", readInt8(384097)); // weight int8
// console.log("mul_ptr[1724416] = ", readInt32Signed(1724416)); // mul int32
// console.log("shift_ptr[1756576] = ", readInt32Signed(1756576)); // mul int32

// camada 65 SOFTMAX
// console.log("op_type = ", readInt32Signed(1800276));
// console.log("act = ", readInt32Signed(1800280));
// console.log("flags = ", readInt32Signed(1800284));
// console.log("in_ptr = " , readInt32Signed(1800288));
// console.log("out_ptr = ", readInt32Signed(1800292));
// console.log("in_h = ", readInt32Signed(1800296));
// console.log("in_w = ", readInt32Signed(1800300));
// console.log("cin = ", readInt32Signed(1800304));
// console.log("cout = ", readInt32Signed(1800308));
// console.log("kh = ", readInt32Signed(1800312));
// console.log("kw = ", readInt32Signed(1800316));
// console.log("stride_h = ", readInt32Signed(1800320));
// console.log("stride_w = ", readInt32Signed(1800324));
// console.log("dil_h = ", readInt32Signed(1800328));
// console.log("dil_w = ", readInt32Signed(1800332));
// console.log("pad_t = ", readInt32Signed(1800336));
// console.log("pad_b = ", readInt32Signed(1800340));
// console.log("pad_l = ", readInt32Signed(1800344));
// console.log("pad_r = ", readInt32Signed(1800348));
// console.log("wptr = ", readInt32Signed(1800352));
// console.log("bias_ptr = ", readInt32Signed(1800356));
// console.log("mul_ptr = ", readInt32Signed(1800360));
// console.log("shift_ptr = ", readInt32Signed(1800364));
// console.log("q6_ptr = ", readInt32Signed(1800368));
// console.log("zx = ", readInt32Signed(1800372));
// console.log("zw = ", readInt32Signed(1800376));
// console.log("zy = ", readInt32Signed(1800380));
// console.log("out_h = ", readInt32Signed(1800384));
// console.log("out_w = ", readInt32Signed(1800388));


// QUANTIZE ULTIMA
// console.log("op_type = ", readInt32Signed(1800392));
// console.log("act = ", readInt32Signed(1800396));
// console.log("flags = ", readInt32Signed(1800400));
// console.log("in_ptr = " , readInt32Signed(1800404));
// console.log("out_ptr = ", readInt32Signed(1800408));
// console.log("in_h = ", readInt32Signed(1800412));
// console.log("in_w = ", readInt32Signed(1800416));
// console.log("cin = ", readInt32Signed(1800420));
// console.log("cout = ", readInt32Signed(1800424));
// console.log("kh = ", readInt32Signed(1800428));
// console.log("kw = ", readInt32Signed(1800432));
// console.log("stride_h = ", readInt32Signed(1800436));
// console.log("stride_w = ", readInt32Signed(1800440));
// console.log("dil_h = ", readInt32Signed(1800444));
// console.log("dil_w = ", readInt32Signed(1800448));
// console.log("pad_t = ", readInt32Signed(1800452));
// console.log("pad_b = ", readInt32Signed(1800456));
// console.log("pad_l = ", readInt32Signed(1800460));
// console.log("pad_r = ", readInt32Signed(1800464));
// console.log("wptr = ", readInt32Signed(1800468));
// console.log("bias_ptr = ", readInt32Signed(1800472));
// console.log("mul_ptr = ", readInt32Signed(1800476));
// console.log("shift_ptr = ", readInt32Signed(1800480));
// console.log("q6_ptr = ", readInt32Signed(1800484));
// console.log("zx = ", readInt32Signed(1800488));
// console.log("zw = ", readInt32Signed(1800492));
// console.log("zy = ", readInt32Signed(1800496));
// console.log("out_h = ", readInt32Signed(1800500));
// console.log("out_w = ", readInt32Signed(1800504));

// CONV2D
// console.log("0 -> ", "op_type = ", readInt32Signed(1792852));
// console.log("1 -> ", "act = ", readInt32Signed(1792856));
// console.log("2 -> ", "flags = ", readInt32Signed(1792860));
// console.log("3 -> ", "in_ptr = " , readInt32Signed(1792864));
// console.log("4 -> ", "out_ptr = ", readInt32Signed(1792868));
// console.log("5 -> ", "in_h = ", readInt32Signed(1792872));
// console.log("6 -> ", "in_w = ", readInt32Signed(1792876));
// console.log("7 -> ", "cin = ", readInt32Signed(1792880));
// console.log("8 -> ", "cout = ", readInt32Signed(1792884));
// console.log("9 -> ", "kh = ", readInt32Signed(1792888));
// console.log("10 ->", "kw = ", readInt32Signed(1792892));
// console.log("11 ->", "stride_h = ", readInt32Signed(1792896));
// console.log("12 ->", "stride_w = ", readInt32Signed(1792900));
// console.log("13 ->", "dil_h = ", readInt32Signed(1792904));
// console.log("14 ->", "dil_w = ", readInt32Signed(1792908));
// console.log("15 ->", "pad_t = ", readInt32Signed(1792912));
// console.log("16 ->", "pad_b = ", readInt32Signed(1792916));
// console.log("17 ->", "pad_l = ", readInt32Signed(1792920));
// console.log("18 ->", "pad_r = ", readInt32Signed(1792924));
// console.log("19 ->", "wptr = ", readInt32Signed(1792928));
// console.log("20 ->", "bias_ptr = ", readInt32Signed(1792932));
// console.log("21 ->", "mul_ptr = ", readInt32Signed(1792936));
// console.log("22 ->", "shift_ptr = ", readInt32Signed(1792940));
// console.log("23 ->", "q6_ptr = ", readInt32Signed(1792944));
// console.log("24 ->", "zx = ", readInt32Signed(1792948));
// console.log("25 ->", "zw = ", readInt32Signed(1792952));
// console.log("26 ->", "zy = ", readInt32Signed(1792956));
// console.log("27 ->", "out_h = ", readInt32Signed(1792960));
// console.log("28 ->", "out_w = ", readInt32Signed(1792964));

// 0 ->  op_type =  1
// 1 ->  act =  3
// 2 ->  flags =  3        
// 3 ->  in_ptr =  1800512 
// 4 ->  out_ptr =  2402624
// 5 ->  in_h =  224       
// 6 ->  in_w =  224
// 7 ->  cin =  3
// 8 ->  cout =  16
// 9 ->  kh =  3
// 10 -> kw =  3
// 11 -> stride_h =  2
// 12 -> stride_w =  2
// 13 -> dil_h =  1
// 14 -> dil_w =  1
// 15 -> pad_t =  0
// 16 -> pad_b =  1
// 17 -> pad_l =  0
// 18 -> pad_r =  1
// 19 -> wptr =  2048
// 20 -> bias_ptr =  1664096
// 21 -> mul_ptr =  1696256
// 22 -> shift_ptr =  1728416
// 23 -> q6_ptr =  1760576
// 24 -> zx =  -1
// 25 -> zw =  0
// 26 -> zy =  -128
// 27 -> out_h =  112
// 28 -> out_w =  112

// DEPTHWISECONV2D
// console.log("0 -> ", "op_type = ", readInt32Signed(1792968));
// console.log("1 -> ", "act = ", readInt32Signed(1792972));
// console.log("2 -> ", "flags = ", readInt32Signed(1792976));
// console.log("3 -> ", "in_ptr = " , readInt32Signed(1792980));
// console.log("4 -> ", "out_ptr = ", readInt32Signed(1792984));
// console.log("5 -> ", "in_h = ", readInt32Signed(1792988));
// console.log("6 -> ", "in_w = ", readInt32Signed(1792992));
// console.log("7 -> ", "cin = ", readInt32Signed(1792996));
// console.log("8 -> ", "cout = ", readInt32Signed(1793000));
// console.log("9 -> ", "kh = ", readInt32Signed(1793004));
// console.log("10 ->", "kw = ", readInt32Signed(1793008));
// console.log("11 ->", "stride_h = ", readInt32Signed(1793012));
// console.log("12 ->", "stride_w = ", readInt32Signed(1793016));
// console.log("13 ->", "dil_h = ", readInt32Signed(1793020));
// console.log("14 ->", "dil_w = ", readInt32Signed(1793024));
// console.log("15 ->", "pad_t = ", readInt32Signed(1793028));
// console.log("16 ->", "pad_b = ", readInt32Signed(1793032));
// console.log("17 ->", "pad_l = ", readInt32Signed(1793036));
// console.log("18 ->", "pad_r = ", readInt32Signed(1793040));
// console.log("19 ->", "wptr = ", readInt32Signed(1793044));
// console.log("20 ->", "bias_ptr = ", readInt32Signed(1793048));
// console.log("21 ->", "mul_ptr = ", readInt32Signed(1793052));
// console.log("22 ->", "shift_ptr = ", readInt32Signed(1793056));
// console.log("23 ->", "q6_ptr = ", readInt32Signed(1793060));
// console.log("24 ->", "zx = ", readInt32Signed(1793064));
// console.log("25 ->", "zw = ", readInt32Signed(1793068));
// console.log("26 ->", "zy = ", readInt32Signed(1793072));
// console.log("27 ->", "out_h = ", readInt32Signed(1793076));
// console.log("28 ->", "out_w = ", readInt32Signed(1793080));

// 0 ->  op_type =  2
// 1 ->  act =  3
// 2 ->  flags =  3
// 3 ->  in_ptr =  2402624
// 4 ->  out_ptr =  3004736
// 5 ->  in_h =  112
// 6 ->  in_w =  112
// 7 ->  cin =  16
// 8 ->  cout =  16
// 9 ->  kh =  3
// 10 -> kw =  3
// 11 -> stride_h =  1
// 12 -> stride_w =  1
// 13 -> dil_h =  1
// 14 -> dil_w =  1
// 15 -> pad_t =  1
// 16 -> pad_b =  1
// 17 -> pad_l =  1
// 18 -> pad_r =  1
// 19 -> wptr =  2480
// 20 -> bias_ptr =  1664160
// 21 -> mul_ptr =  1696320
// 22 -> shift_ptr =  1728480
// 23 -> q6_ptr =  1760640
// 24 -> zx =  -128
// 25 -> zw =  0
// 26 -> zy =  -128
// 27 -> out_h =  112
// 28 -> out_w =  112

// ADD
console.log("0 -> ", "op_type = ", readInt32Signed(1793896));
console.log("1 -> ", "act = ", readInt32Signed(1793900));
console.log("2 -> ", "flags = ", readInt32Signed(1793904));
console.log("3 -> ", "in_ptr = " , readInt32Signed(1793908));
console.log("4 -> ", "out_ptr = ", readInt32Signed(1793912));
console.log("5 -> ", "in_h = ", readInt32Signed(1793916));
console.log("6 -> ", "in_w = ", readInt32Signed(1793920));
console.log("7 -> ", "cin = ", readInt32Signed(1793924));
console.log("8 -> ", "cout = ", readInt32Signed(1793928));
console.log("9 -> ", "kh = ", readInt32Signed(1793932));
console.log("10 ->", "kw = ", readInt32Signed(1793936));
console.log("11 ->", "stride_h = ", readInt32Signed(1793940));
console.log("12 ->", "stride_w = ", readInt32Signed(1793944));
console.log("13 ->", "dil_h = ", readInt32Signed(1793948));
console.log("14 ->", "dil_w = ", readInt32Signed(1793952));
console.log("15 ->", "pad_t = ", readInt32Signed(1793956));
console.log("16 ->", "pad_b = ", readInt32Signed(1793960));
console.log("17 ->", "pad_l = ", readInt32Signed(1793964));
console.log("18 ->", "pad_r = ", readInt32Signed(1793968));
console.log("19 ->", "wptr = ", readInt32Signed(1793972));
console.log("20 ->", "bias_ptr = ", readInt32Signed(1793976));
console.log("21 ->", "mul_ptr = ", readInt32Signed(1793980));
console.log("22 ->", "shift_ptr = ", readInt32Signed(1793984));
console.log("23 ->", "q6_ptr = ", readInt32Signed(1793988));
console.log("24 ->", "zx = ", readInt32Signed(1793992));
console.log("25 ->", "zw = ", readInt32Signed(1793996));
console.log("26 ->", "zy = ", readInt32Signed(1794000));
console.log("27 ->", "out_h = ", readInt32Signed(1794004));
console.log("28 ->", "out_w = ", readInt32Signed(1794008));

// 0 ->  op_type =  4
// 1 ->  act =  0
// 2 ->  flags =  0
// 3 ->  in_ptr =  1800512
// 4 ->  out_ptr =  3004736
// 5 ->  in_h =  56
// 6 ->  in_w =  56
// 7 ->  cin =  8
// 8 ->  cout =  8
// 9 ->  kh =  1400234221
// 10 -> kw =  -1
// 11 -> stride_h =  1073741824
// 12 -> stride_w =  0
// 13 -> dil_h =  1073741824
// 14 -> dil_w =  2
// 15 -> pad_t =  1800512
// 16 -> pad_b =  2402624
// 17 -> pad_l =  -3
// 18 -> pad_r =  -11
// 19 -> wptr =  2048
// 20 -> bias_ptr =  0
// 21 -> mul_ptr =  0
// 22 -> shift_ptr =  0
// 23 -> q6_ptr =  0
// 24 -> zx =  -3
// 25 -> zw =  -11
// 26 -> zy =  -11
// 27 -> out_h =  56
// 28 -> out_w =  56

// MEAN
console.log("0 -> ", "op_type = ", readInt32Signed(1800044));
console.log("1 -> ", "act = ", readInt32Signed(1800048));
console.log("2 -> ", "flags = ", readInt32Signed(1800052));
console.log("3 -> ", "in_ptr = " , readInt32Signed(1800056));
console.log("4 -> ", "out_ptr = ", readInt32Signed(1800060));
console.log("5 -> ", "in_h = ", readInt32Signed(1800064));
console.log("6 -> ", "in_w = ", readInt32Signed(1800068));
console.log("7 -> ", "cin = ", readInt32Signed(1800072));
console.log("8 -> ", "cout = ", readInt32Signed(1800076));
console.log("9 -> ", "kh = ", readInt32Signed(1800080));
console.log("10 ->", "kw = ", readInt32Signed(1800084));
console.log("11 ->", "stride_h = ", readInt32Signed(1800088));
console.log("12 ->", "stride_w = ", readInt32Signed(1800092));
console.log("13 ->", "dil_h = ", readInt32Signed(1800096));
console.log("14 ->", "dil_w = ", readInt32Signed(1800100));
console.log("15 ->", "pad_t = ", readInt32Signed(1800104));
console.log("16 ->", "pad_b = ", readInt32Signed(1800108));
console.log("17 ->", "pad_l = ", readInt32Signed(1800112));
console.log("18 ->", "pad_r = ", readInt32Signed(1800116));
console.log("19 ->", "wptr = ", readInt32Signed(1800120));
console.log("20 ->", "bias_ptr = ", readInt32Signed(1800124));
console.log("21 ->", "mul_ptr = ", readInt32Signed(1800128));
console.log("22 ->", "shift_ptr = ", readInt32Signed(1800132));
console.log("23 ->", "q6_ptr = ", readInt32Signed(1800136));
console.log("24 ->", "zx = ", readInt32Signed(1800140));
console.log("25 ->", "zw = ", readInt32Signed(1800144));
console.log("26 ->", "zy = ", readInt32Signed(1800148));
console.log("27 ->", "out_h = ", readInt32Signed(1800152));
console.log("28 ->", "out_w = ", readInt32Signed(1800156));

// 0 ->  op_type =  5
// 1 ->  act =  0
// 2 ->  flags =  0
// 3 ->  in_ptr =  1800512
// 4 ->  out_ptr =  2402624
// 5 ->  in_h =  7
// 6 ->  in_w =  7
// 7 ->  cin =  1280
// 8 ->  cout =  1280
// 9 ->  kh =  1079542796
// 10 -> kw =  1
// 11 -> stride_h =  49
// 12 -> stride_w =  1
// 13 -> dil_h =  1
// 14 -> dil_w =  1
// 15 -> pad_t =  1800512
// 16 -> pad_b =  0
// 17 -> pad_l =  0
// 18 -> pad_r =  0
// 19 -> wptr =  2048
// 20 -> bias_ptr =  0
// 21 -> mul_ptr =  0
// 22 -> shift_ptr =  0
// 23 -> q6_ptr =  0
// 24 -> zx =  -128
// 25 -> zw =  0
// 26 -> zy =  -128
// 27 -> out_h =  1
// 28 -> out_w =  1

// QUANTIZE L0
console.log("============================");
console.log("QUANTIZE L0");
console.log("0 -> ", "op_type = ", readInt32Signed(1792736));
console.log("1 -> ", "act = ", readInt32Signed(1792740));
console.log("2 -> ", "flags = ", readInt32Signed(1792744));
console.log("3 -> ", "in_ptr = " , readInt32Signed(1792748));
console.log("4 -> ", "out_ptr = ", readInt32Signed(1792752));
console.log("5 -> ", "in_h = ", readInt32Signed(1792756));
console.log("6 -> ", "in_w = ", readInt32Signed(1792760));
console.log("7 -> ", "cin = ", readInt32Signed(1792764));
console.log("8 -> ", "cout = ", readInt32Signed(1792768));
console.log("9 -> ", "kh = ", readInt32Signed(1792772));
console.log("10 ->", "kw = ", readInt32Signed(1792776));
console.log("11 ->", "stride_h = ", readInt32Signed(1792780));
console.log("12 ->", "stride_w = ", readInt32Signed(1792784));
console.log("13 ->", "dil_h = ", readInt32Signed(1792788));
console.log("14 ->", "dil_w = ", readInt32Signed(1792792));
console.log("15 ->", "pad_t = ", readInt32Signed(1792796));
console.log("16 ->", "pad_b = ", readInt32Signed(1792800));
console.log("17 ->", "pad_l = ", readInt32Signed(1792804));
console.log("18 ->", "pad_r = ", readInt32Signed(1792808));
console.log("19 ->", "wptr = ", readInt32Signed(1792812));
console.log("20 ->", "bias_ptr = ", readInt32Signed(1792816));
console.log("21 ->", "mul_ptr = ", readInt32Signed(1792820));
console.log("22 ->", "shift_ptr = ", readInt32Signed(1792824));
console.log("23 ->", "q6_ptr = ", readInt32Signed(1792828));
console.log("24 ->", "zx = ", readInt32Signed(1792832));
console.log("25 ->", "zw = ", readInt32Signed(1792836));
console.log("26 ->", "zy = ", readInt32Signed(1792840));
console.log("27 ->", "out_h = ", readInt32Signed(1792844));
console.log("28 ->", "out_w = ", readInt32Signed(1792848));

// QUANTIZE L0
// 0 ->  op_type =  7
// 1 ->  act =  0
// 2 ->  flags =  0
// 3 ->  in_ptr =  1800512
// 4 ->  out_ptr =  1800512
// 5 ->  in_h =  224
// 6 ->  in_w =  224
// 7 ->  cin =  3
// 8 ->  cout =  3
// 9 ->  kh =  1073741824
// 10 -> kw =  1
// 11 -> stride_h =  0
// 12 -> stride_w =  0
// 13 -> dil_h =  1
// 14 -> dil_w =  1
// 15 -> pad_t =  1800512
// 16 -> pad_b =  0
// 17 -> pad_l =  0
// 18 -> pad_r =  0
// 19 -> wptr =  2048
// 20 -> bias_ptr =  0
// 21 -> mul_ptr =  0
// 22 -> shift_ptr =  0
// 23 -> q6_ptr =  0
// 24 -> zx =  127
// 25 -> zw =  0
// 26 -> zy =  -1
// 27 -> out_h =  224
// 28 -> out_w =  224

///////////////////////////////////////////////////
// 0 ->  op_type =  7
// 1 ->  act =  0
// 2 ->  flags =  0
// 3 ->  in_ptr =  1800512
// 4 ->  out_ptr =  1800512
// 5 ->  in_h =  1
// 6 ->  in_w =  1
// 7 ->  cin =  1000
// 8 ->  cout =  1000
// 9 ->  kh =  1073741824
// 10 -> kw =  1
// 11 -> stride_h =  0
// 12 -> stride_w =  0
// 13 -> dil_h =  1
// 14 -> dil_w =  1
// 15 -> pad_t =  1800512
// 16 -> pad_b =  0
// 17 -> pad_l =  0
// 18 -> pad_r =  0
// 19 -> wptr =  2048
// 20 -> bias_ptr =  0
// 21 -> mul_ptr =  0
// 22 -> shift_ptr =  0
// 23 -> q6_ptr =  0
// 24 -> zx =  -128
// 25 -> zw =  0
// 26 -> zy =  0
// 27 -> out_h =  1
// 28 -> out_w =  1

// op_type =  6
// act =  0
// flags =  0
// in_ptr =  3004736
// out_ptr =  1800512
// in_h =  1
// in_w =  1
// cin =  1000
// cout =  1000
// kh =  1318294144
// kw =  -8
// stride_h =  -128
// stride_w =  5
// dil_h =  1
// dil_w =  1
// pad_t =  3004736
// pad_b =  0
// pad_l =  0
// pad_r =  0
// wptr =  2048
// bias_ptr =  0
// mul_ptr =  0
// shift_ptr =  0
// q6_ptr =  0
// zx =  -45
// zw =  0
// zy =  -128
// out_h =  1
// out_w =  1




// console.log("===================")
// debugAddress(1692256);
// console.log("-------------------")
// console.log("bias_ptr[1692256] = ", readRaw4(1692256)); // weight int8
// console.log("+++++++++++++++++++++++++++++++++++++++++")



// 11101101 11011101 11101010 00100010
// 0xED     0xDD     0xEA     0x22
// op_type =  3
// act =  0
// flags =  0
// > in_ptr =  2402624
// > out_ptr =  3004736
// in_h =  1
// in_w =  1
// > cin =  1280
// > cout =  1000
// kh =  1
// kw =  1
// stride_h =  1
// stride_w =  1
// dil_h =  1
// dil_w =  1
// pad_t =  0
// pad_b =  0
// pad_l =  0
// pad_r =  0
// wptr =  384096
// bias_ptr =  1692256
// mul_ptr =  1724416
// shift_ptr =  1756576
// q6_ptr =  0
// zx =  -128
// zw =  0
// zy =  -45
// out_h =  1
// out_w =  1

// ================== L64 ==================
// op_index          : 64
// optype            : FULLY_CONNECTED
// op_type           : 3 (FC)
// act               : 0 (NONE)
// flags             : 0 (0)
// in_slot/out_slot  : 1 -> 2
// in_ptr/out_ptr    : 2402624 -> 3004736
// in_h/in_w         : 1 x 1
// cin/cout          : 1280 -> 1000
// kh/kw             : 1 x 1
// stride_h/stride_w : 1 x 1
// dil_h/dil_w       : 1 x 1
// pad t/b/l/r       : 0 0 0 0
// out_h/out_w       : 1 x 1
// w_off/b_off       : 382048 / 28160
// mul_off/q6_off     : 28160 / 28160
// shift_off           : 28160
// shift_ptr           : 1756576
// wptr/bias/mul/q6   : 384096 / 1692256 / 1724416 / 0
// zx/zw/zy           : -128 / 0 / -45

////////////////////////////////////

//  ;; ================== L65 ==================
//  ;; op_index          : 65
//  ;; optype            : SOFTMAX
//  ;; op_type           : 6 (SOFTMAX)
//  ;; act               : 0 (NONE)
//  ;; flags             : 0 (0)
//  ;; in_slot/out_slot  : 2 -> 0
//  ;; in_ptr/out_ptr    : 3004736 -> 1800512
//  ;; --- SOFTMAX Quantization ---
//  ;; sX/sY              : 0.076735 / 0.003906
//  ;; zX/zY              : -45 / -128
//  ;; beta               : 1.000000
//  ;; integer_bits       : 5
//  ;; internal_scale     : 0.031250
//  ;; input_beta_mul     : 1318294144
//  ;; input_beta_left_sh : -8
//  ;; diff_min           : -128
//  ;; in_h/in_w         : 1 x 1
//  ;; cin/cout          : 1000 -> 1000
//  ;; kh/kw (βmul/shft) : 1318294144 / -8
//  ;; stride_h (diffMin): -128
//  ;; stride_w (intBits): 5
//  ;; pad_t (inPtr)     : 3004736
//  ;; out_h/out_w       : 1 x 1
//  ;; w_off/b_off       : 0 / 0
//  ;; mul_off/q6_off     : 0 / 0
//  ;; shift_off           : 0
//  ;; shift_ptr           : 0
//  ;; wptr/bias/mul/q6   : 2048 / 0 / 0 / 0
//  ;; zx/zw/zy           : -45 / 0 / -128
//  ;;