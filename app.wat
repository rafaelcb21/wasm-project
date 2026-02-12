;; ============================================================
;; ADD Layer - Soma elemento a elemento de 2 tensores
;; Para ADD: pad_t, pad_b contêm os ponteiros dos inputs
;; ============================================================
(func $add (export "add") (param $layer_idx i32)
  (local $base     i32)
  (local $op_type  i32)
  (local $in_ptr_0 i32)  ;; primeiro input (de pad_t)
  (local $in_ptr_1 i32)  ;; segundo input (de pad_b)
  (local $out_ptr  i32)
  (local $in_h     i32)
  (local $in_w     i32)
  (local $cin      i32)
  (local $zx       i32)
  (local $zy       i32)
  
  (local $total    i32)
  (local $idx      i32)
  (local $val0     i32)
  (local $val1     i32)
  (local $sum      i32)
  (local $result   i32)

  ;; Carregar base
  local.get $layer_idx
  call $layerparam_base
  local.set $base

  ;; Carregar ponteiros de input dos campos pad_t e pad_b
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

  ;; zx (23) - zero point do input
  local.get $base
  i32.const 23
  i32.const 2
  i32.shl
  i32.add
  i32.load align=4
  local.set $zx

  ;; zy (25) - zero point do output
  local.get $base
  i32.const 25
  i32.const 2
  i32.shl
  i32.add
  i32.load align=4
  local.set $zy

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

      ;; val0 = load<i8>(in_ptr_0 + idx)
      local.get $in_ptr_0
      local.get $idx
      i32.add
      i32.load8_s
      local.set $val0

      ;; val1 = load<i8>(in_ptr_1 + idx)
      local.get $in_ptr_1
      local.get $idx
      i32.add
      i32.load8_s
      local.set $val1

      ;; sum = (val0 - zx) + (val1 - zx) + zy
      local.get $val0
      local.get $zx
      i32.sub
      local.get $val1
      local.get $zx
      i32.sub
      i32.add
      local.get $zy
      i32.add
      local.set $sum

      ;; Clamp [-128, 127]
      local.get $sum
      i32.const 127
      i32.gt_s
      (if
        (then
          i32.const 127
          local.set $sum
        )
      )

      local.get $sum
      i32.const -128
      i32.lt_s
      (if
        (then
          i32.const -128
          local.set $sum
        )
      )

      ;; store<i8>(out_ptr + idx) = sum
      local.get $out_ptr
      local.get $idx
      i32.add
      local.get $sum
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
;; MEAN Layer - Média espacial (Global Average Pooling)
;; Calcula média sobre H x W para cada canal
;; ============================================================
(func $mean (export "mean") (param $layer_idx i32)
  (local $base     i32)
  (local $in_ptr   i32)
  (local $out_ptr  i32)
  (local $in_h     i32)
  (local $in_w     i32)
  (local $cin      i32)
  (local $zx       i32)
  (local $zy       i32)
  
  (local $spatial_size i32)
  (local $c        i32)
  (local $idx      i32)
  (local $sum      i32)
  (local $val      i32)
  (local $mean_val i32)

  ;; Carregar base
  local.get $layer_idx
  call $layerparam_base
  local.set $base

  ;; in_ptr (3) - para MEAN vem de pad_t
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

  ;; zx (23)
  local.get $base
  i32.const 23
  i32.const 2
  i32.shl
  i32.add
  i32.load align=4
  local.set $zx

  ;; zy (25)
  local.get $base
  i32.const 25
  i32.const 2
  i32.shl
  i32.add
  i32.load align=4
  local.set $zy

  ;; spatial_size = in_h * in_w
  local.get $in_h
  local.get $in_w
  i32.mul
  local.set $spatial_size

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

          ;; val = load<i8>(in_ptr + (idx * cin + c))
          local.get $in_ptr
          local.get $idx
          local.get $cin
          i32.mul
          local.get $c
          i32.add
          i32.add
          i32.load8_s
          local.set $val

          ;; sum += (val - zx)
          local.get $sum
          local.get $val
          local.get $zx
          i32.sub
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

      ;; mean_val = (sum / spatial_size) + zy
      local.get $sum
      local.get $spatial_size
      i32.div_s
      local.get $zy
      i32.add
      local.set $mean_val

      ;; Clamp [-128, 127]
      local.get $mean_val
      i32.const 127
      i32.gt_s
      (if
        (then
          i32.const 127
          local.set $mean_val
        )
      )

      local.get $mean_val
      i32.const -128
      i32.lt_s
      (if
        (then
          i32.const -128
          local.set $mean_val
        )
      )

      ;; store<i8>(out_ptr + c) = mean_val
      local.get $out_ptr
      local.get $c
      i32.add
      local.get $mean_val
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
;; SOFTMAX Layer - Implementação simplificada para int8
;; ============================================================
(func $softmax (export "softmax") (param $layer_idx i32)
  (local $base     i32)
  (local $in_ptr   i32)
  (local $out_ptr  i32)
  (local $cin      i32)
  (local $zx       i32)
  (local $zy       i32)
  
  (local $i        i32)
  (local $max_val  i32)
  (local $val      i32)
  (local $sum      i32)
  (local $exp_val  i32)
  (local $out_val  i32)

  ;; Carregar base
  local.get $layer_idx
  call $layerparam_base
  local.set $base

  ;; in_ptr (de pad_t para SOFTMAX)
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

  ;; zx (23)
  local.get $base
  i32.const 23
  i32.const 2
  i32.shl
  i32.add
  i32.load align=4
  local.set $zx

  ;; zy (25)
  local.get $base
  i32.const 25
  i32.const 2
  i32.shl
  i32.add
  i32.load align=4
  local.set $zy

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

  ;; Para int8 quantizado, softmax simplificado:
  ;; Apenas copiar os valores (softmax real requer exponenciais)
  ;; Em produção, isso seria implementado com lookup tables
  i32.const 0
  local.set $i

  (block $exit_copy
    (loop $loop_copy
      local.get $i
      local.get $cin
      i32.ge_s
      br_if $exit_copy

      ;; Simplesmente copiar (para quantizado, softmax é aproximado)
      local.get $in_ptr
      local.get $i
      i32.add
      i32.load8_s
      local.set $val

      local.get $out_ptr
      local.get $i
      i32.add
      local.get $val
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