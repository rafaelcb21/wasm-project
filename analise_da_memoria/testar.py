INT32_MIN = -2147483648
INT32_MAX =  2147483647

def to_int32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x

def arshift_int32(x: int, n: int) -> int:
    # shift aritmético em int32
    x = to_int32(x)
    return to_int32(x >> n)

#def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
#    a = to_int32(a)
#    b = to_int32(b)
#
#    if a == INT32_MIN and b == INT32_MIN:
#        return INT32_MAX
#
#    ab = int(a) * int(b)
#
#    # nudge exatamente como no C++
#    if ab >= 0:
#        nudge = (1 << 30)
#    else:
#        nudge = (1 << 30) - 1
#
#    ab += nudge
#
#    # força comportamento int64 com sinal antes do shift
#    ab = (ab + (1 << 63)) % (1 << 64) - (1 << 63)
#
#    result = ab >> 31
#
#    return to_int32(result)

#def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
#    a = to_int32(a)
#    b = to_int32(b)
#
#    if a == INT32_MIN and b == INT32_MIN:
#        return INT32_MAX
#
#    # Multiplicação em 64-bit
#    ab = a * b
#    
#    # O segredo: o arredondamento "doubling high mul" 
#    # Em muitas implementações, o nudge é fixo para 1 << 30
#    nudge = 1 << 30
#    return to_int32((ab + nudge) >> 31)

def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    # Caso especial de estouro do int32
    if a == -2147483648 and b == -2147483648:
        return 2147483647

    # 1. Multiplicação full precision (64-bit)
    ab = a * b
    
    # 2. O arredondamento (nudge)
    # No padrão Gemmlowp/TFLite, se ab for positivo, somamos 1<<30.
    # Se for negativo, somamos 1<<30, mas o comportamento do shift à direita 
    # no Python para negativos é "floor", então precisamos compensar.
    nudge = 1 << 30
    
    # 3. Double e High Mul
    # (ab + nudge) >> 31 simula o retorno dos 32 bits altos de (2 * a * b)
    result = (ab + nudge) >> 31
    
    return to_int32(result)

#def rounding_divide_by_pot(x: int, exponent: int) -> int:
#    x = to_int32(x)
#    if exponent <= 0: return x
#
#    # Voltando para a lógica simétrica pura:
#    nudge = 1 << (exponent - 1)
#    result = arshift_int32(x + nudge, exponent)
#    
#    return to_int32(result)

#def rounding_divide_by_pot(x: int, exponent: int) -> int:
#    x = to_int32(x)
#    if exponent <= 0:
#        return x
#
#    mask = (1 << exponent) - 1  # 511 para exponent = 9
#    remainder = x & mask        # 46336 & 511 = 256
#    threshold = (mask >> 1)     # 511 / 2^1 = 255
#    if x < 0:
#        threshold += 1
#
#    result = arshift_int32(x, exponent)
#
#    if remainder > threshold:
#        result += 1
#    print(x, exponent, result, to_int32(result))
#    return to_int32(result)

#def rounding_divide_by_pot(x: int, exponent: int) -> int:
#    x = to_int32(x)
#    if exponent <= 0:
#        return x
#    
#    # O valor que define o "meio" para arredondamento (ex: 2^(9-1) = 256)
#    nudge = 1 << (exponent - 1)
#    
#    # Adicionamos o nudge e fazemos o shift. 
#    # O shift aritmético (arshift_int32) já cuida do sinal corretamente.
#    result = arshift_int32(x + nudge, exponent)
#    
#    # Remova o print para performance, ou use para debug:
#    # print(f"x: {x}, exp: {exponent}, result: {result}")
#    
#    return to_int32(result)

#def rounding_divide_by_pot(x: int, exponent: int) -> int:
#    x = to_int32(x)
#    if exponent <= 0:
#        return x
#
#    # Para x = 46336 e exp = 9:
#    # mask = 511
#    mask = (1 << exponent) - 1
#    
#    # remainder = 46336 & 511 = 256
#    remainder = x & mask
#    
#    # threshold = 255
#    threshold = (mask >> 1)
#    
#    # Se x for negativo, o threshold sobe para 256
#    if x < 0:
#        threshold += 1
#
#    result = arshift_int32(x, exponent)
#
#    # Se 256 > 255: result + 1
#    # PARA DAR 90: Precisamos mudar a condição para >= ou ajustar o threshold
#    if remainder > threshold:
#        result += 1
#        
#    return to_int32(result)

def rounding_divide_by_pot(x: int, exponent: int) -> int:
    # Simula exatamente o comportamento de: (x + (1 << (exponent - 1)) - (x < 0 ? 0 : 1)) >> exponent
    # Isso faz com que o 0.5 positivo arredonde para baixo e o 0.5 negativo para cima (ou vice-versa)
    nudge = 1 << (exponent - 1)
    if x >= 0:
        # Se você precisa que 90.5 vire 90, subtraímos 1 do nudge
        return arshift_int32(x + nudge - 1, exponent)
    else:
        return arshift_int32(x + nudge, exponent)

def multiply_by_quantized_multiplier(x: int, multiplier: int, shift: int) -> int:
    x = to_int32(x)
    x = saturating_rounding_doubling_high_mul(x, multiplier)
    return rounding_divide_by_pot(x, -shift)

def conv2d():
    """
    params: dict com todos os campos do layer
    memory: bytearray representando a memória linear do WASM
    """
    with open("conv1_input.txt") as f:
        input = [int(line.strip()) for line in f if line.strip()]

    bias = [61676, -30175, 50681, -1073741824, 57799, -1073741824, 87029, 1073741824, 96241, 62237, 78015, -1073741824, 24002, -1073741824, 64138, -38688]
    mul = [1164940590, 1403857037, 1222486747, 1508056806, 1246209585, 1092111233, 1262063410, 1758890566, 1458035258, 1384968417, 1710028868, 1129778733, 1943334477, 2052087036, 1083663744, 1458238180]
    shift = [-8, -7, -8, -23, -8, -23, -9, -27, -9, -7, -9, -23, -8, -26, -8, -7]
    q6 = [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127]
    weights = [15, 30, 9, -65, -127, -38, 53, 105, 23, 15, 34, 10, -66, -120, -34, 60, 105, 23, -1, 2, 4, -19, -22, -10, 18, 10, 6, -16, 3, -42, 31, 72, 45, 107, 127, 86, -6, 10, -9, -7, -26, 15, 36, 3, 10, 2, 19, -20, 4, 17, 4, 13, 16, 13, -49, -85, -22, -69, -127, -33, -25, -53, -14, 17, 53, 14, 17, 61, 14, 14, 30, 8, 34, 56, 16, 50, 84, 24, 28, 22, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 111, 28, -47, -91, -23, 4, 8, -2, 68, 127, 31, -54, -96, -24, 5, 13, 0, 27, 23, 12, -17, -18, -5, -3, -3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, -7, -18, 81, -32, -52, 72, -19, -56, 64, -12, -55, 127, -44, -87, 62, -15, -45, 61, -8, -56, 64, -13, -50, -18, 13, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, -20, -3, -5, -13, -12, 9, -20, 5, -64, -103, -44, -66, -127, -42, -6, -26, -11, -42, -46, -8, -31, -24, -5, 10, 7, 4, -64, 111, 18, -28, 103, -55, 75, -41, -29, -23, 28, 8, 4, 48, -36, 127, -93, -27, 5, -49, 35, 14, -9, 2, 69, -80, 15, -7, 25, -17, -39, 84, -44, -30, 60, -30, -24, 57, -32, -66, 127, -64, -25, 53, -30, -16, 42, -26, -30, 54, -28, 13, -25, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -31, -55, -18, -45, -78, -19, -16, -25, -8, 42, 96, 24, 64, 127, 29, 32, 40, 13, -19, -50, -12, -28, -61, -12, -9, -13, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -17, -31, -1, 60, 111, 33, 10, 7, 6, -14, -27, -3, 72, 127, 34, 13, 11, 8, -4, 0, -1, 23, 28, 12, 5, 6, 4, 17, 45, 6, 2, -25, 2, -2, 6, 3, -124, -127, -28, -81, -51, -8, 29, -1, -2, -39, -105, -66, -15, -42, -15, 22, 2, 2]
    memory = []

    # Parametros
    op_type =  1
    act =  3
    flags =  3
    in_ptr =  0 # 1800512
    out_ptr =  2402624
    in_h =  224
    in_w =  224
    cin =  3
    cout =  16
    kh =  3
    kw =  3
    stride_h =  2
    stride_w =  2
    dil_h =  1
    dil_w =  1
    pad_t =  0
    pad_b =  1
    pad_l =  0
    pad_r =  1
    wptr =  2048
    bias_ptr =  1664096
    mul_ptr =  1696256
    shift_ptr =  1728416
    q6_ptr =  1760576
    zx =  -1
    zw =  0
    zy =  -128
    out_h =  112
    out_w =  112

    bottom = pad_t + in_h
    right = pad_l + in_w
    plane_out = out_h * out_w
    w_per_oc = kh * kw * cin


    # -------- ORDEM CORRETA NHWC --------

    for i_out in range(out_h):
        i = i_out * stride_h

        for j_out in range(out_w):
            j = j_out * stride_w

            for oc in range(cout):

                b = bias[oc]
                m = mul[oc]
                s = shift[oc]
                q6_ = q6[oc]

                acc = b

                for ki in range(kh):
                    row = i + ki * dil_h

                    if row < pad_t:
                        continue
                    if row >= bottom:
                        break

                    row_img = row - pad_t
                    if row_img < 0 or row_img >= in_h:
                        continue

                    row_base = row_img * in_w

                    for kj in range(kw):
                        col = j + kj * dil_w

                        if col < pad_l:
                            continue
                        if col >= right:
                            break

                        col_img = col - pad_l
                        if col_img < 0 or col_img >= in_w:
                            continue

                        for c in range(cin):
                            idx = (row_base + col_img) * cin + c
                            tmp = input[idx]

                            pos = (ki * kw + kj) * cin + c
                            w0 = weights[oc * w_per_oc + pos]

                            acc += (tmp - zx) * (w0 - zw)

                y = multiply_by_quantized_multiplier(acc, m, s)
                y += zy

                # RELU6
                lo = zy
                hi = q6_

                if hi < lo:
                    hi = lo
                if hi > 127:
                    hi = 127
                if lo < -128:
                    lo = -128

                if y < lo:
                    y = lo
                if y > hi:
                    y = hi

                if y > 127:
                    y = 127
                if y < -128:
                    y = -128

                memory.append(y)

    return memory


result = conv2d()

with open("meu_output.txt", "w") as f:
    for value in result:
        f.write(f"{value}\n")

print("Arquivo gerado com sucesso.")