import csv
from pathlib import Path
from collections import defaultdict, deque
import tflite
import tflite.Model as TFLModel
import re
import struct
import numpy as np
import sys
import math

MODEL_PATH = "mobilenetv2_alpha035_quant.tflite"
OUT_WAT_PATH = "tensors_mul_q6_slots_params_alpha035.wat"

KERNEL_BASE_HINT = 2048
ALIGN = 16
BATCH = 1
RSHIFT_DEFAULT = 31
NUM_SLOTS = 3

# ===== setup flatc modules path (Colab) =====
sys.path.insert(0, "/content")

import tflite.Conv2DOptions as Conv2DOptions
import tflite.DepthwiseConv2DOptions as DepthwiseConv2DOptions
import tflite.FullyConnectedOptions as FullyConnectedOptions
import tflite.AddOptions as AddOptions
import tflite.Padding as Padding
import tflite.ActivationFunctionType as ActivationFunctionType

tensor_type_map = {
    0: ('float32', np.float32),
    1: ('float16', np.float16),
    2: ('int32', np.int32),
    3: ('uint8', np.uint8),
    4: ('int64', np.int64),
    6: ('bool', np.bool_),
    7: ('int16', np.int16),
    9: ('int8', np.int8),
}
BYTES_PER_TYPE = {0:4, 1:2, 2:4, 3:1, 4:8, 6:1, 7:2, 9:1}

# ---------- helpers ----------
# Domínio real
#LUT_MIN = -10.0
#LUT_MAX = 0.0
#STEP    = 0.01
#
#LUT_LEN  = int((LUT_MAX - LUT_MIN) / STEP) + 1
#LUT_BASE = 0
#
#def build_exp_lut_real(min_x=-10.0, max_x=0.0, step=0.01):
#    out = bytearray()
#    x = min_x
#    for _ in range(LUT_LEN):
#        val = math.exp(x)
#        out += struct.pack("<f", float(val))  # float32 little endian
#        x += step
#    return bytes(out)
#
#lut_blob = build_exp_lut_real(LUT_MIN, LUT_MAX, STEP)

def align_up(x, a=16):
    return (x + (a - 1)) & ~(a - 1)

def wat_data_from_bytes(raw: bytes, base_address: int) -> str:
    wat_str = '"' + ''.join(f'\\{b:02x}' for b in raw) + '"'
    return f'(data (i32.const {base_address}) {wat_str})'

def qparams_np(t):
    q = t.Quantization()
    if q is None:
        return None
    s = q.ScaleAsNumpy()
    z = q.ZeroPointAsNumpy()
    if s is None:
        return None
    s = np.atleast_1d(np.array(s, dtype=np.float64))
    z = np.atleast_1d(np.array(z if z is not None else [], dtype=np.int64))
    if s.size == 0:
        return None
    return dict(scales=s, zps=z, qdim=q.QuantizedDimension())

def scale_scalar(t):
    """Extrai escala scalar de um tensor"""
    q = t.Quantization()
    if q is None:
        return 1.0
    s = q.ScaleAsNumpy()
    if s is None or len(s) == 0:
        return 1.0
    return float(np.array(s, dtype=np.float64).flatten()[0])

def zp_scalar(t):
    q = t.Quantization()
    if q is None:
        return 0
    z = q.ZeroPointAsNumpy()
    if z is None or len(z) == 0:
        return 0
    return int(np.array(z, dtype=np.int64).flatten()[0])

def op_name(model, op):
    code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    for k, v in tflite.BuiltinOperator.__dict__.items():
        if isinstance(v, int) and v == code:
            return k
    return "CUSTOM"

def safe_bytes_from_tensor(model, subgraph, tensor_idx):
    tensor = subgraph.Tensors(int(tensor_idx))
    buffer = model.Buffers(tensor.Buffer())
    try:
        data_bytes = buffer.DataAsNumpy()
    except AttributeError:
        return None, None, None
    if not hasattr(data_bytes, "__len__") or len(data_bytes) == 0:
        return None, None, None

    shape = tensor.ShapeAsNumpy()
    dtype = int(tensor.Type())
    _, npdt = tensor_type_map.get(dtype, (None, None))
    if npdt is None:
        return None, None, None

    arr = np.frombuffer(data_bytes.tobytes(), dtype=npdt)
    try:
        arr = arr.reshape(shape)
    except Exception:
        pass

    raw = arr.flatten().tobytes()
    return tensor, arr, raw

def is_constant_tensor(model, sub, tidx):
    t = sub.Tensors(tidx)
    b = model.Buffers(t.Buffer())
    try:
        data = b.DataAsNumpy()
    except AttributeError:
        return False
    return hasattr(data, "__len__") and len(data) > 0

def tensor_numel(shape):
    n = 1
    for d in shape:
        d = int(d)
        if d < 0:
            d = BATCH
        n *= d
    return n

def tensor_shape_list(t):
    s = t.ShapeAsNumpy()
    if s is None:
        return []
    return [int(x) for x in s.tolist()]

def same_padding(in_h, in_w, k_h, k_w, s_h, s_w, dil_h=1, dil_w=1):
    out_h = (in_h + s_h - 1) // s_h
    out_w = (in_w + s_w - 1) // s_w
    eff_kh = (k_h - 1) * dil_h + 1
    eff_kw = (k_w - 1) * dil_w + 1

    pad_h_total = max(0, (out_h - 1) * s_h + eff_kh - in_h)
    pad_w_total = max(0, (out_w - 1) * s_w + eff_kw - in_w)

    pad_t = pad_h_total // 2
    pad_b = pad_h_total - pad_t
    pad_l = pad_w_total // 2
    pad_r = pad_w_total - pad_l
    return pad_t, pad_b, pad_l, pad_r, out_h, out_w

def resolve_slot_from_producer(tid, visiting=None):
    """Resolve slot de tensor seguindo cadeia de produtores"""
    if visiting is None:
        visiting = set()
    if tid in visiting:
        return None  # ciclo
    visiting.add(tid)

    # Já mapeado?
    if tid in tensor_to_slot:
        return tensor_to_slot[tid]

    # Tem produtor?
    p = producer_by_tensor.get(tid)
    if p is None:
        return None

    # Produtor é útil?
    if p in old_idx_to_label:
        lbl = old_idx_to_label[p]
        for alloc in slot_allocation:
            if alloc['layer'] == lbl:
                tensor_to_slot[tid] = alloc['output_slot']
                return alloc['output_slot']

    # Produtor ignorado: herda de inputs
    op_p = subgraph.Operators(p)
    for j in range(op_p.InputsLength()):
        in_tid = int(op_p.Inputs(j))
        if in_tid < 0 or is_constant_tensor(model, subgraph, in_tid):
            continue
        s = resolve_slot_from_producer(in_tid, visiting.copy())
        if s is not None:
            tensor_to_slot[tid] = s
            return s

    return None

# ---------- enums we store in params ----------
ACT_NONE = 0
ACT_RELU = 1
ACT_RELU6 = 3

OP_CONV = 1
OP_DW = 2
OP_FC = 3
OP_ADD = 4
OP_MEAN = 5
OP_SOFTMAX = 6
OP_QUANTIZE = 7

FLAG_PADDING_SAME = 1 << 0
FLAG_HAS_Q6 = 1 << 1

LP_FMT = "<" + "i"*29
LP_SIZE = struct.calcsize(LP_FMT)

def parse_fused_activation(faf_int: int) -> int:
    try:
        if faf_int == ActivationFunctionType.NONE:
            return ACT_NONE
        if faf_int == ActivationFunctionType.RELU:
            return ACT_RELU
        if faf_int == ActivationFunctionType.RELU6:
            return ACT_RELU6
    except Exception:
        pass
    if faf_int == 0: return ACT_NONE
    if faf_int == 1: return ACT_RELU
    if faf_int == 3: return ACT_RELU6
    return ACT_NONE

def _padding_is_same(pad_int: int) -> bool:
    try:
        return pad_int == Padding.SAME
    except Exception:
        return pad_int == 0

def parse_conv2d_options(op):
    try:
        u = op.BuiltinOptions()
        if hasattr(u, "Bytes") and hasattr(u, "Pos"):
            opt = Conv2DOptions.Conv2DOptions() if hasattr(Conv2DOptions, "Conv2DOptions") else Conv2DOptions()
            opt.Init(u.Bytes, u.Pos)
            stride_h = int(opt.StrideH())
            stride_w = int(opt.StrideW())
            try:
                dil_h = int(opt.DilationHFactor())
                dil_w = int(opt.DilationWFactor())
            except Exception:
                dil_h, dil_w = 1, 1
            pad_kind = 0 if _padding_is_same(int(opt.Padding())) else 1
            act = parse_fused_activation(int(opt.FusedActivationFunction()))
            return stride_h, stride_w, dil_h, dil_w, pad_kind, act
    except Exception:
        pass
    return 1, 1, 1, 1, 1, ACT_NONE

def parse_dwconv2d_options(op):
    try:
        u = op.BuiltinOptions()
        if hasattr(u, "Bytes") and hasattr(u, "Pos"):
            opt = DepthwiseConv2DOptions.DepthwiseConv2DOptions() if hasattr(DepthwiseConv2DOptions, "DepthwiseConv2DOptions") else DepthwiseConv2DOptions()
            opt.Init(u.Bytes, u.Pos)
            stride_h = int(opt.StrideH())
            stride_w = int(opt.StrideW())
            try:
                dil_h = int(opt.DilationHFactor())
                dil_w = int(opt.DilationWFactor())
            except Exception:
                dil_h, dil_w = 1, 1
            depth_mult = int(opt.DepthMultiplier())
            pad_kind = 0 if _padding_is_same(int(opt.Padding())) else 1
            act = parse_fused_activation(int(opt.FusedActivationFunction()))
            return stride_h, stride_w, dil_h, dil_w, pad_kind, act, depth_mult
    except Exception:
        pass
    return 1, 1, 1, 1, 1, ACT_NONE, 1

def parse_fc_options(op):
    try:
        u = op.BuiltinOptions()
        if hasattr(u, "Bytes") and hasattr(u, "Pos"):
            opt = FullyConnectedOptions.FullyConnectedOptions() if hasattr(FullyConnectedOptions, "FullyConnectedOptions") else FullyConnectedOptions()
            opt.Init(u.Bytes, u.Pos)
            return parse_fused_activation(int(opt.FusedActivationFunction()))
    except Exception:
        pass
    return ACT_NONE

def parse_add_options(op):
    """Extrai activation function de ADD"""
    try:
        u = op.BuiltinOptions()
        if hasattr(u, "Bytes") and hasattr(u, "Pos"):
            opt = AddOptions.AddOptions() if hasattr(AddOptions, "AddOptions") else AddOptions()
            opt.Init(u.Bytes, u.Pos)
            return parse_fused_activation(int(opt.FusedActivationFunction()))
    except Exception:
        pass
    return ACT_NONE

# ============================================================
# QUANTIZAÇÃO - Funções corrigidas com frexp
# ============================================================

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

def quantize_multiplier(real_multiplier: float):
    """
    Decompõe real_multiplier em (q31_multiplier, shift) via frexp.
    Relação: real_multiplier ~= q31_multiplier * 2^(shift - 31)

    Convenção usada aqui (TFLM-style):
      shift > 0 => LEFT SHIFT (multiplicação por 2^shift)
      shift < 0 => RIGHT SHIFT (divisão por 2^(-shift))
    """
    rm = float(real_multiplier)
    if rm == 0.0:
        return 0, 0

    q, exp = math.frexp(rm)  # q in [0.5, 1), exp é o expoente
    q31 = int(round(q * (1 << 31)))

    if q31 == (1 << 31):
        q31 //= 2
        exp += 1

    if q31 > INT32_MAX:
        q31 = INT32_MAX
    if q31 < INT32_MIN:
        q31 = INT32_MIN

    return int(q31), int(exp)

def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    """Multiplicação Q31 com saturação e arredondamento"""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = int(a) * int(b)
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    res = (ab + nudge) // (1 << 31)
    if res > INT32_MAX:
        return INT32_MAX
    if res < INT32_MIN:
        return INT32_MIN
    return int(res)

def rounding_divide_by_pot(x: int, exponent: int) -> int:
    """Divisão com arredondamento por potência de 2"""
    if exponent <= 0:
        return int(x)
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1)
    if x < 0:
        threshold += 1
    return (x >> exponent) + (1 if remainder > threshold else 0)

def multiply_by_quantized_multiplier(x: int, multiplier: int, shift: int) -> int:
    """
    Referência para runtime:
      shift > 0: left shift antes do mul
      shift < 0: right shift depois do mul
    """
    x = int(x)
    multiplier = int(multiplier)
    shift = int(shift)

    if shift > 0:
        x = x * (1 << shift)

    x = saturating_rounding_doubling_high_mul(x, multiplier)

    if shift < 0:
        x = rounding_divide_by_pot(x, -shift)

    return int(x)

def compute_add_quantization_params(sA, sB, sY):
    """
    Calcula parâmetros de quantização para ADD segundo TFLite.

    Fórmula TFLite:
    1. Escolher escala comum: s_common = max(sA, sB) * 2
    2. mul0 = sA / s_common, shift0
    3. mul1 = sB / s_common, shift1
    4. out_mul = s_common / sY, out_shift

    Retorna: (mul0, shift0, mul1, shift1, out_mul, out_shift, s_common)
    """
    # Escala comum (2x a maior escala de entrada)
    s_common = max(sA, sB) * 2.0

    if sA == 0.0 and sB == 0.0:
        return 0, 0, 0, 0, 0, 0, s_common
    if sY == 0.0:
        return 0, 0, 0, 0, 0, 0, s_common
    if s_common == 0.0:
        return 0, 0, 0, 0, 0, 0, s_common

    # Input 0: sA -> s_common
    ratio0 = sA / s_common
    mul0, shift0 = quantize_multiplier(ratio0)

    # Input 1: sB -> s_common
    ratio1 = sB / s_common
    mul1, shift1 = quantize_multiplier(ratio1)

    out_ratio = s_common / sY
    out_mul, out_shift = quantize_multiplier(out_ratio)

    return mul0, shift0, mul1, shift1, out_mul, out_shift, s_common

def pack_layerparam(
    op_type, act, flags,
    in_ptr, out_ptr,
    in_h, in_w, cin, cout,
    kh, kw, stride_h, stride_w,
    dil_h, dil_w,
    pad_t, pad_b, pad_l, pad_r,
    wptr, bias_ptr, mul_ptr, shift_ptr, q6_ptr,
    zx, zw, zy, out_h, out_w
):
    return struct.pack(
        LP_FMT,
        op_type, act, flags,
        in_ptr, out_ptr,
        in_h, in_w, cin, cout,
        kh, kw,
        stride_h, stride_w,
        dil_h, dil_w,
        pad_t, pad_b, pad_l, pad_r,
        wptr, bias_ptr, mul_ptr, shift_ptr, q6_ptr,
        zx, zw, zy, out_h, out_w
    )

# ---------- load model ----------
buf = Path(MODEL_PATH).read_bytes()
if hasattr(TFLModel, "GetRootAsModel"):
    model = TFLModel.GetRootAsModel(buf, 0)
elif hasattr(TFLModel, "Model") and hasattr(TFLModel.Model, "GetRootAsModel"):
    model = TFLModel.Model.GetRootAsModel(buf, 0)
else:
    raise RuntimeError("Binding tflite.Model não possui GetRootAsModel.")

subgraph = model.Subgraphs(0)

# ============================================================
# EXTRAIR GRAFO E GERAR STRING DATA
# ============================================================

def build_graph_for_subgraph(model, sg):
    n_ops = sg.OperatorsLength()
    producer_by_tensor = {}
    consumers_by_tensor = defaultdict(list)

    op_types = []
    for op_idx in range(n_ops):
        op = sg.Operators(op_idx)
        typ = op_name(model, op)
        op_types.append(typ)

        for j in range(op.OutputsLength()):
            tid = int(op.Outputs(j))
            if tid >= 0:
                producer_by_tensor[tid] = op_idx

        for j in range(op.InputsLength()):
            tid = int(op.Inputs(j))
            if tid >= 0:
                consumers_by_tensor[tid].append(op_idx)

    return op_types, producer_by_tensor, consumers_by_tensor

#def compute_useful_adjacency(sg, op_types, consumers_by_tensor, ignored_types={"QUANTIZE"}):
def compute_useful_adjacency(sg, op_types, consumers_by_tensor, ignored_types=set()):  # ← REMOVIDO {"QUANTIZE"}
    n_ops = sg.OperatorsLength()
    ignored = set(i for i, t in enumerate(op_types) if t in ignored_types)
    useful = [i for i in range(n_ops) if i not in ignored]

    forward = defaultdict(set)
    for op_idx in range(n_ops):
        op = sg.Operators(op_idx)
        for j in range(op.OutputsLength()):
            tid = int(op.Outputs(j))
            if tid < 0:
                continue
            for c in consumers_by_tensor.get(tid, []):
                if c != op_idx:
                    forward[op_idx].add(c)

    backward = defaultdict(set)
    for src, dsts in forward.items():
        for dst in dsts:
            backward[dst].add(src)

    def next_useful_from(op_idx):
        result = set()
        stack = [op_idx]
        seen = set()
        while stack:
            cur = stack.pop()
            for nxt in forward.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt in ignored:
                    stack.append(nxt)
                else:
                    result.add(nxt)
        return result

    def prev_useful_to(op_idx):
        result = set()
        stack = [op_idx]
        seen = set()
        while stack:
            cur = stack.pop()
            for prv in backward.get(cur, []):
                if prv in seen:
                    continue
                seen.add(prv)
                if prv in ignored:
                    stack.append(prv)
                else:
                    result.add(prv)
        return result

    useful_inputs = {u: set() for u in useful}
    useful_outputs = {u: set() for u in useful}

    for u in useful:
        useful_inputs[u] = prev_useful_to(u)
        useful_outputs[u] = next_useful_from(u)

    return useful, useful_inputs, useful_outputs

def topo_order(nodes, in_edges, out_edges):
    indeg = {n: len(in_edges[n]) for n in nodes}
    q = deque(sorted([n for n in nodes if indeg[n] == 0]))
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(out_edges[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(nodes):
        raise RuntimeError("Grafo possui ciclo (inesperado para TFLite).")
    return order

# Gerar string data
data_lines = ["nome_da_camada; camada_atual; camada_acima; camada_de_baixo"]

op_types, producer_by_tensor, consumers_by_tensor = build_graph_for_subgraph(model, subgraph)
useful, useful_inputs, useful_outputs = compute_useful_adjacency(
    subgraph, op_types, consumers_by_tensor, ignored_types=set()
)

new_label = {old_idx: f"L{i}" for i, old_idx in enumerate(useful)}
order = topo_order(useful, useful_inputs, useful_outputs)

for old_idx in order:
    in_labels = sorted([new_label[p] for p in useful_inputs[old_idx] if p in new_label],
                      key=lambda x: int(x[1:]))
    out_labels = sorted([new_label[n] for n in useful_outputs[old_idx] if n in new_label],
                       key=lambda x: int(x[1:]))

    in_str = "[" + ", ".join(in_labels) + "]" if in_labels else "[]"
    out_str = "[" + ", ".join(out_labels) + "]" if out_labels else "[]"

    line = f"{op_types[old_idx]}; {new_label[old_idx]}; {in_str}; {out_str}"
    data_lines.append(line)

data = "\n".join(data_lines)

print("=" * 80)
print("STRING DATA GERADA:")
print("=" * 80)
print(data)
print("=" * 80)

# ============================================================
# ALOCAR SLOTS USANDO ALGORITMO CORRETO
# ============================================================

def parse_layers(data):
    layers = []
    lines = data.strip().split('\n')[1:]

    for line in lines:
        parts = [p.strip() for p in line.split(';')]
        layer_type = parts[0]
        layer_name = parts[1]

        def parse_layer_list(s):
            if not s or s == '[]':
                return []
            matches = re.findall(r'L\d+', s)
            return matches

        layers_above = parse_layer_list(parts[2])
        layers_below = parse_layer_list(parts[3])

        layers.append({
            'type': layer_type,
            'name': layer_name,
            'above': layers_above,
            'below': layers_below
        })

    return layers

def allocate_slots(layers):
    layer_output_slot = {}
    slot_readers_count = {}
    allocation = []
    next_slot = 1

    for i, layer in enumerate(layers):
        layer_name = layer['name']
        layer_type = layer['type'] 
        layers_above = layer['above']
        layers_below = layer['below']

        # ========== NOVO: QUANTIZE usa mesmo slot do input ==========
        if layer_type == 'QUANTIZE':
            # QUANTIZE não aloca slot novo, usa o do input
            if not layers_above:
                input_slot = 0
            else:
                input_slot = layer_output_slot[layers_above[0]]
            
            # Output slot = input slot (operação in-place)
            layer_output_slot[layer_name] = input_slot
            
            allocation.append({
                'layer': layer_name,
                'type': layer_type,
                'input_slots': [input_slot],
                'output_slot': input_slot,  # ← MESMO SLOT!
            })
            
            print(f"{layer_type:25} {layer_name:5} [{input_slot} -> {input_slot}] (in-place)")
            continue  # ← Pula o resto da lógica
        # ============================================================

        if not layers_above:
            input_slots = [0]
            next_slot = 1
        else:
            input_slots = [layer_output_slot[above] for above in layers_above]

        available_slots = list(range(NUM_SLOTS))

        for slot in list(available_slots):
            if slot in slot_readers_count and slot_readers_count[slot] > 0:
                available_slots.remove(slot)

        # --- NOVO: proteção contra overwrite silencioso ---
        if not available_slots:
            raise RuntimeError(f"Sem slots livres em {layer_name}")

        if next_slot in available_slots:
            output_slot = next_slot
        else:
            output_slot = available_slots[0] if available_slots else next_slot

        for input_slot in input_slots:
            if input_slot in slot_readers_count:
                slot_readers_count[input_slot] -= 1
                if slot_readers_count[input_slot] == 0:
                    del slot_readers_count[input_slot]

        if layers_below:
            slot_readers_count[output_slot] = len(layers_below)

        layer_output_slot[layer_name] = output_slot

        allocation.append({
            'layer': layer_name,
            'type': layer_type,
            'input_slots': input_slots,
            'output_slot': output_slot,
        })

        next_slot = (output_slot + 1) % NUM_SLOTS

    return allocation, layer_output_slot

layers = parse_layers(data)
slot_allocation, layer_output_slot = allocate_slots(layers)

print("\n" + "=" * 80)
print("ALOCAÇÃO DE SLOTS:")
print("=" * 80)
for alloc in slot_allocation:
    ins = alloc['input_slots']
    outs = alloc['output_slot']
    if len(ins) == 1:
        print(f"{alloc['type']:25} {alloc['layer']:5} [{ins[0]} -> {outs}]")
    else:
        print(f"{alloc['type']:25} {alloc['layer']:5} [{' e '.join(map(str, ins))} -> {outs}]")

# ============================================================
# CRIAR MAPEAMENTO TENSOR_ID → SLOT (CORREÇÃO CRÍTICA)
# ============================================================

print("\n" + "=" * 80)
print("CRIANDO MAPEAMENTO TENSOR_ID → SLOT:")
print("=" * 80)

# Mapeamento reverso: label → op_idx
label_to_op_idx = {label: op_idx for op_idx, label in new_label.items()}

# Mapeamento direto: op_idx original -> label (Lx)
old_idx_to_label = {old_idx: label for old_idx, label in new_label.items()}


# Criar mapeamento tensor_id → slot
tensor_to_slot = {}

for alloc in slot_allocation:
    label = alloc['layer']
    op_idx = label_to_op_idx.get(label)

    if op_idx is None:
        continue

    op = subgraph.Operators(op_idx)

    # Mapear todos os outputs desta operação para o slot de saída
    for j in range(op.OutputsLength()):
        tensor_id = int(op.Outputs(j))
        if tensor_id >= 0:
            tensor_to_slot[tensor_id] = alloc['output_slot']
            print(f"  tensor {tensor_id} (produzido por {label}) → slot {alloc['output_slot']}")

graph_inputs = set(int(subgraph.Inputs(i)) for i in range(subgraph.InputsLength()))

for tidx in range(subgraph.TensorsLength()):
    if tidx in tensor_to_slot:
        continue
    if is_constant_tensor(model, subgraph, tidx):
        continue

    if tidx in graph_inputs:
        tensor_to_slot[tidx] = 0
        print(f"  tensor {tidx} (input do subgrafo) → slot 0")
    else:
        # NÃO force slot 0 para intermediário desconhecido
        print(f"  [PEND] tensor {tidx} não mapeado (intermediário)")


# Aplicar fechamento
for tidx in range(subgraph.TensorsLength()):
    if tidx not in tensor_to_slot and not is_constant_tensor(model, subgraph, tidx):
        resolve_slot_from_producer(tidx)

print(f"Total de tensores mapeados (após fechamento): {len(tensor_to_slot)}")

# Checagem pós-fechamento (diagnóstico)
unmapped_after = []
for tidx in range(subgraph.TensorsLength()):
    if is_constant_tensor(model, subgraph, tidx):
        continue
    if tidx not in tensor_to_slot:
        unmapped_after.append(tidx)

if unmapped_after:
    print(f"[WARN] tensores sem slot após fechamento: {unmapped_after}")
else:
    print("✅ Fechamento de mapeamento: nenhum tensor não-constante pendente")

# ============================================================
# VALIDAÇÃO FORTE: nenhum tensor usado por op útil pode ficar sem slot
# ============================================================
for i in range(subgraph.OperatorsLength()):
    # Só valida ops úteis (as que realmente entram no pipeline)
    if i not in old_idx_to_label:
        continue

    op = subgraph.Operators(i)

    # Inputs não-constantes precisam estar mapeados
    for j in range(op.InputsLength()):
        tid = int(op.Inputs(j))
        if tid < 0:
            continue
        if is_constant_tensor(model, subgraph, tid):
            continue
        if tid not in tensor_to_slot:
            raise RuntimeError(
                f"[MAP-ERROR] input tensor sem slot: op_index={i}, tensor_id={tid}, op={op_name(model, op)}"
            )

    # Outputs também precisam estar mapeados
    for j in range(op.OutputsLength()):
        tid = int(op.Outputs(j))
        if tid < 0:
            continue
        if tid not in tensor_to_slot:
            raise RuntimeError(
                f"[MAP-ERROR] output tensor sem slot: op_index={i}, tensor_id={tid}, op={op_name(model, op)}"
            )

print("✅ Validação de mapeamento tensor_to_slot: OK")



# ============================================================
# (A) EXPORT WEIGHTS/BIAS
# ============================================================
weights_raw = bytearray()
bias_raw = bytearray()
weight_tensor_off = {}
bias_tensor_off = {}

for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    optype = op_name(model, op)
    if optype not in ("CONV_2D","DEPTHWISE_CONV_2D","FULLY_CONNECTED"):
        continue

    in_ids = [int(k) for k in op.InputsAsNumpy() if int(k) >= 0]
    if len(in_ids) < 2:
        continue

    w_id = in_ids[1]
    _, arrw, raww = safe_bytes_from_tensor(model, subgraph, w_id)
    if arrw is not None and w_id not in weight_tensor_off:
        weight_tensor_off[w_id] = len(weights_raw)
        weights_raw += raww

    if len(in_ids) >= 3:
        b_id = in_ids[2]
        _, arrb, rawb = safe_bytes_from_tensor(model, subgraph, b_id)
        if arrb is not None and arrb.ndim == 1 and b_id not in bias_tensor_off:
            bias_tensor_off[b_id] = len(bias_raw)
            bias_raw += rawb

# ============================================================
# (B) EXTRACT MUL + Q6 (Conv/DW/FC)
# ============================================================
mul_vals = []
shift_vals = []
q6_vals = []
mul_q6_off = {}

for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    optype = op_name(model, op)

    if optype not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
        continue

    in_ids = [int(k) for k in op.InputsAsNumpy() if int(k) >= 0]
    out_ids = [int(k) for k in op.OutputsAsNumpy() if int(k) >= 0]
    if len(in_ids) < 2 or len(out_ids) < 1:
        continue

    in0 = subgraph.Tensors(in_ids[0])
    wts = subgraph.Tensors(in_ids[1])
    out0 = subgraph.Tensors(out_ids[0])

    w_shape = tensor_shape_list(wts)

    if optype == "CONV_2D":
        nfeat = int(w_shape[0]) if len(w_shape) >= 1 else None
    elif optype == "DEPTHWISE_CONV_2D":
        nfeat = int(w_shape[3]) if len(w_shape) >= 4 else None
    else:
        nfeat = int(w_shape[0]) if len(w_shape) >= 1 else None

    if nfeat is None:
        continue

    qx = qparams_np(in0)
    qw = qparams_np(wts)
    qy = qparams_np(out0)
    if qx is None or qw is None or qy is None:
        continue
    if qx["scales"].size == 0 or qw["scales"].size == 0 or qy["scales"].size == 0:
        continue

    sx = float(qx["scales"][0])
    sw_arr = qw["scales"].astype(np.float64)
    sy_arr = qy["scales"].astype(np.float64)
    sy = float(sy_arr[0])

    mul_off = len(mul_vals) * 4
    q6_off = len(q6_vals) * 4

    if sw_arr.size == 1:
        rm = (float(sw_arr[0]) * sx) / sy
        mul, shift = quantize_multiplier(rm)
        mul_vals.extend([mul] * nfeat)
        shift_vals.extend([shift] * nfeat)
    else:
        use = min(nfeat, sw_arr.size)
        rmv = (sw_arr[:use] * sx) / sy
        for rm in rmv:
            mul, shift = quantize_multiplier(rm)
            mul_vals.append(int(mul))
            shift_vals.append(int(shift))

        if nfeat > use:
            mul_vals.extend([mul_vals[-1]] * (nfeat - use))
            shift_vals.extend([shift_vals[-1]] * (nfeat - use))

        if nfeat > use:
            mul_vals.extend([mul_vals[-1]] * (nfeat - use))
            shift_vals.extend([shift_vals[-1]] * (nfeat - use))

    zy = zp_scalar(out0)

    if sy_arr.size == 1:
        q6 = int(np.round(6.0 / float(sy_arr[0]))) + zy
        q6_vals.extend([q6] * nfeat)
    else:
        use = min(nfeat, sy_arr.size)
        q6v = (np.round(6.0 / sy_arr[:use]).astype(np.int64) + np.int64(zy))
        q6_vals.extend(int(x) for x in q6v)
        if nfeat > use:
            q6_vals.extend([int(q6v[-1])] * (nfeat - use))

    shift_off = len(shift_vals) * 4 - nfeat * 4

    mul_q6_off[i] = (mul_off, shift_off, q6_off, nfeat)

mul_blob = np.array(mul_vals, dtype="<i4").tobytes()
shift_blob = np.array(shift_vals, dtype="<i4").tobytes()

q6_blob = np.array(q6_vals, dtype="<i4").tobytes()

# ============================================================
# (C) SLOT_BYTES (max tensor não-constante)
# ============================================================
max_bytes = 0
max_info = None

for tidx in range(subgraph.TensorsLength()):
    t = subgraph.Tensors(tidx)
    if is_constant_tensor(model, subgraph, tidx):
        continue

    shape = tensor_shape_list(t)
    if not shape:
        continue

    ttype = int(t.Type())
    bpe = BYTES_PER_TYPE.get(ttype)
    if bpe is None:
        continue

    nbytes = tensor_numel(shape) * bpe
    if nbytes > max_bytes:
        max_bytes = nbytes
        max_info = (tidx, t.Name().decode("utf-8", "ignore"), shape, ttype, nbytes)

SLOT_BYTES = align_up(max_bytes, ALIGN)

# ============================================================
# (D) bases
# ============================================================
kernel_base = align_up(KERNEL_BASE_HINT, ALIGN)
kernel_bytes = len(weights_raw)

bias_base = align_up(kernel_base + kernel_bytes, ALIGN)
bias_bytes = len(bias_raw)

mul_base = align_up(bias_base + bias_bytes, ALIGN)
mul_bytes = len(mul_blob)

shift_base = align_up(mul_base + mul_bytes, ALIGN)
shift_bytes = len(shift_blob)

q6_base = align_up(shift_base + shift_bytes, ALIGN)
q6_bytes = len(q6_blob)

params_base = align_up(q6_base + q6_bytes, ALIGN)


# ============================================================
# (E) CRIAR LAYER PARAMS COM SLOTS CORRETOS E QUANTIZAÇÃO
# ============================================================

# Mapeamento de label (L0, L1...) para slot
label_to_slot = {alloc['layer']: alloc['output_slot'] for alloc in slot_allocation}
label_to_input_slots = {alloc['layer']: alloc['input_slots'] for alloc in slot_allocation}

layer_params = []

# ============================================================
# CALCULAR SLOT_BASES CORRETO (antes de criar layer_params)
# ============================================================

# Primeiro, contar quantas layers teremos (excluindo QUANTIZE que já está em useful)
num_layers = len([i for i in range(subgraph.OperatorsLength()) if i in old_idx_to_label])

params_bytes = align_up(num_layers * LP_SIZE, ALIGN)

slot0_base = align_up(params_base + params_bytes, ALIGN)
slot_bases = [slot0_base]
for _ in range(1, NUM_SLOTS):
    slot_bases.append(align_up(slot_bases[-1] + SLOT_BYTES, ALIGN))

print(f"\n[INFO] Calculando slot_bases ANTES de criar layer_params:")
print(f"  num_layers estimado: {num_layers}")
print(f"  params_bytes: {params_bytes}")
print(f"  slot_bases: {slot_bases}")
print()

# ============================================================
# (E) CRIAR LAYER PARAMS COM SLOTS CORRETOS E QUANTIZAÇÃO
# ============================================================

for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    optype = op_name(model, op)

    # Verificar se é operação útil (incluindo QUANTIZE agora)
    if i not in old_idx_to_label:
        continue

    label = old_idx_to_label[i]
    out_slot = label_to_slot.get(label, 0)
    in_slots = label_to_input_slots.get(label, [0])

    # Determinar op_type
    if optype == "CONV_2D":
        op_type = OP_CONV
    elif optype == "DEPTHWISE_CONV_2D":
        op_type = OP_DW
    elif optype == "FULLY_CONNECTED":
        op_type = OP_FC
    elif optype == "ADD":
        op_type = OP_ADD
    elif optype == "MEAN":
        op_type = OP_MEAN
    elif optype == "SOFTMAX":
        op_type = OP_SOFTMAX
    elif optype == "QUANTIZE":
        op_type = OP_QUANTIZE
    else:
        continue  # Ignora outras ops

    in_ids = [int(k) for k in op.InputsAsNumpy() if int(k) >= 0]
    out_ids = [int(k) for k in op.OutputsAsNumpy() if int(k) >= 0]

    if len(out_ids) < 1:
        continue

    # ========== QUANTIZE (Requantização) ==========
    if optype == "QUANTIZE":
        if len(in_ids) < 1:
            continue

        in0 = subgraph.Tensors(in_ids[0])
        out0 = subgraph.Tensors(out_ids[0])

        in_shape = tensor_shape_list(in0)
        out_shape = tensor_shape_list(out0)

        # Dimensões (assumindo formato NHWC ou NC)
        in_h = in_shape[1] if len(in_shape) >= 3 else 1
        in_w = in_shape[2] if len(in_shape) >= 3 else 1
        cin = in_shape[3] if len(in_shape) >= 4 else (in_shape[1] if len(in_shape) == 2 else 1)

        out_h = out_shape[1] if len(out_shape) >= 3 else 1
        out_w = out_shape[2] if len(out_shape) >= 3 else 1
        cout = out_shape[3] if len(out_shape) >= 4 else (out_shape[1] if len(out_shape) == 2 else 1)

        # Parâmetros de quantização
        scale_in = scale_scalar(in0)
        zp_in = zp_scalar(in0)
        scale_out = scale_scalar(out0)
        zp_out = zp_scalar(out0)

        # Calcular multiplicador para conversão: scale_in / scale_out
        if scale_out == 0.0:
            raise RuntimeError(f"QUANTIZE op_index={i}: escala de saída scale_out=0")
        
        ratio = scale_in / scale_out
        mul_quantize, shift_quantize = quantize_multiplier(ratio)

        # Usar tensor_to_slot
        if in_ids[0] not in tensor_to_slot:
            raise RuntimeError(f"QUANTIZE op_index={i}: input tensor sem slot mapeado (tensor_id={in_ids[0]})")
        slot_in = tensor_to_slot[in_ids[0]]
        in_ptr_0 = slot_bases[slot_in]

        print(f"\nQUANTIZE {label}: scale_in={scale_in:.6f}, scale_out={scale_out:.6f}, ratio={ratio:.6f}")
        print(f"  mul={mul_quantize}, shift={shift_quantize}, zp_in={zp_in}, zp_out={zp_out}")

        layer_params.append({
            "op_index": i,
            "optype": optype,
            "op_type": OP_QUANTIZE,
            "act": ACT_NONE,
            "flags": 0,
            "in_slot": slot_in,
            "out_slot": out_slot,
            "in_h": int(in_h), "in_w": int(in_w),
            "cin": int(cin), "cout": int(cout),
            "kh": int(mul_quantize),    # mul (Q31)
            "kw": int(shift_quantize),  # shift
            "stride_h": 0,  # Não usado
            "stride_w": 0,  # Não usado
            "dil_h": 1, "dil_w": 1,
            "pad_t": in_ptr_0,  # input_ptr
            "pad_b": 0, "pad_l": 0, "pad_r": 0,
            "out_h": int(out_h), "out_w": int(out_w),
            "w_off": 0,
            "has_bias": False,
            "b_off": 0,
            "has_mulq6": False,
            "mul_off": 0,
            "q6_off": 0,
            "zx": int(zp_in),   # zero point input
            "zw": 0,
            "zy": int(zp_out),  # zero point output
            "depth_mult": 1,
            "input_slots": [slot_in],
            "input_ptrs": [in_ptr_0],
            "quant_params": {
                "scale_in": float(scale_in),
                "scale_out": float(scale_out),
                "zp_in": int(zp_in),
                "zp_out": int(zp_out),
                "mul": int(mul_quantize),
                "shift": int(shift_quantize),
                "ratio": float(ratio)
            }
        })
        continue

    # ========== ADD COM QUANTIZAÇÃO COMPLETA (CORRIGIDO) ==========
    if optype == "ADD":
        if len(in_ids) < 2:
            continue

        in0 = subgraph.Tensors(in_ids[0])
        in1 = subgraph.Tensors(in_ids[1])
        out0 = subgraph.Tensors(out_ids[0])

        in_shape = tensor_shape_list(in0)
        out_shape = tensor_shape_list(out0)

        in_h = in_shape[1] if len(in_shape) >= 3 else 1
        in_w = in_shape[2] if len(in_shape) >= 3 else 1
        cin = in_shape[3] if len(in_shape) >= 4 else (in_shape[1] if len(in_shape) == 2 else 1)

        out_h = out_shape[1] if len(out_shape) >= 3 else 1
        out_w = out_shape[2] if len(out_shape) >= 3 else 1
        cout = out_shape[3] if len(out_shape) >= 4 else (out_shape[1] if len(out_shape) == 2 else 1)

        # Extrair parâmetros de quantização
        sA = scale_scalar(in0)
        zA = zp_scalar(in0)
        sB = scale_scalar(in1)
        zB = zp_scalar(in1)
        sY = scale_scalar(out0)
        zY = zp_scalar(out0)

        # Calcular multiplicadores e shifts
        mul0, shift0, mul1, shift1, out_mul, out_shift, s_common = compute_add_quantization_params(sA, sB, sY)

        # Extrair activation
        act = parse_add_options(op)

        if in_ids[0] not in tensor_to_slot or in_ids[1] not in tensor_to_slot:
            raise RuntimeError(
                f"ADD op_index={i}: input tensor sem slot mapeado. "
                f"in_ids={in_ids[:2]}"
            )

        slot_A = tensor_to_slot[in_ids[0]]
        slot_B = tensor_to_slot[in_ids[1]]


        # --- NOVO: diagnóstico de divergência entre alocação e mapeamento real ---
        expected = label_to_input_slots.get(label, [])
        if len(expected) == 2 and expected != [slot_A, slot_B]:
            print(f"[WARN] {label}: input_slots alloc={expected} != tensor_to_slot={[slot_A, slot_B]}")

        in_ptr_0 = slot_bases[slot_A]
        in_ptr_1 = slot_bases[slot_B]

        # Verificação de debug
        print(f"\nADD {label}: tensor_ids={in_ids[:2]}, slots=[{slot_A}, {slot_B}], ptrs=[{in_ptr_0}, {in_ptr_1}]")

        layer_params.append({
            "op_index": i,
            "optype": optype,
            "op_type": op_type,
            "act": act,
            "flags": 0,
            "in_slot": slot_A,
            "out_slot": out_slot,
            "in_h": int(in_h), "in_w": int(in_w),
            "cin": int(cin), "cout": int(cout),
            "kh": int(mul0),      # mul0 (Q31)
            "kw": int(shift0),    # shift0
            "stride_h": int(mul1),# mul1 (Q31)
            "stride_w": int(shift1), # shift1
            "dil_h": int(out_mul),   # out_mul (Q31)
            "dil_w": int(out_shift), # out_shift
            "pad_t": in_ptr_0,    # in_ptr[0] - CORRETO via tensor_to_slot
            "pad_b": in_ptr_1,    # in_ptr[1] - CORRETO via tensor_to_slot
            "pad_l": int(zA),     # zero point A
            "pad_r": int(zB),     # zero point B
            "out_h": int(out_h), "out_w": int(out_w),
            "w_off": 0,
            "has_bias": False,
            "b_off": 0,
            "has_mulq6": False,
            "mul_off": 0,
            "q6_off": 0,
            "zx": int(zA), "zw": int(zB), "zy": int(zY),
            "depth_mult": 1,
            "input_slots": [slot_A, slot_B],  # Slots corretos
            "input_ptrs": [in_ptr_0, in_ptr_1],
            "quant_params": {
                "sA": sA, "sB": sB, "sY": sY,
                "zA": zA, "zB": zB, "zY": zY,
                "mul0": mul0, "shift0": shift0,
                "mul1": mul1, "shift1": shift1,
                "out_mul": out_mul, "out_shift": out_shift,
                "s_common": s_common
            }
        })
        continue

    # ========== MEAN COM QUANTIZAÇÃO ==========
    if optype == "MEAN":
        if len(in_ids) < 1:
            continue

        in0 = subgraph.Tensors(in_ids[0])
        out0 = subgraph.Tensors(out_ids[0])

        in_shape = tensor_shape_list(in0)
        out_shape = tensor_shape_list(out0)

        in_h = in_shape[1] if len(in_shape) >= 3 else 1
        in_w = in_shape[2] if len(in_shape) >= 3 else 1
        cin = in_shape[3] if len(in_shape) >= 4 else (in_shape[1] if len(in_shape) == 2 else 1)

        out_h = out_shape[1] if len(out_shape) >= 3 else 1
        out_w = out_shape[2] if len(out_shape) >= 3 else 1
        cout = out_shape[3] if len(out_shape) >= 4 else (out_shape[1] if len(out_shape) == 2 else 1)

        sX = scale_scalar(in0)
        zX = zp_scalar(in0)
        sY = scale_scalar(out0)
        zY = zp_scalar(out0)

        # Para MEAN: multiplier = sX / sY
        if sY == 0.0:
            raise RuntimeError(f"MEAN op_index={i}: escala de saída sY=0")
        ratio = sX / sY
        mul_mean, shift_mean = quantize_multiplier(ratio)


        # Usar tensor_to_slot
        if in_ids[0] not in tensor_to_slot:
            raise RuntimeError(f"MEAN op_index={i}: input tensor sem slot mapeado (tensor_id={in_ids[0]})")
        slot_in = tensor_to_slot[in_ids[0]]
        in_ptr_0 = slot_bases[slot_in]


        layer_params.append({
            "op_index": i,
            "optype": optype,
            "op_type": op_type,
            "act": ACT_NONE,
            "flags": 0,
            "in_slot": slot_in,
            "out_slot": out_slot,
            "in_h": int(in_h), "in_w": int(in_w),
            "cin": int(cin), "cout": int(cout),
            "kh": int(mul_mean),    # mul
            "kw": int(shift_mean),  # shift
            "stride_h": int(in_h * in_w), # spatial_size
            "stride_w": 1,
            "dil_h": 1, "dil_w": 1,
            "pad_t": in_ptr_0,
            "pad_b": 0, "pad_l": 0, "pad_r": 0,
            "out_h": int(out_h), "out_w": int(out_w),
            "w_off": 0,
            "has_bias": False,
            "b_off": 0,
            "has_mulq6": False,
            "mul_off": 0,
            "q6_off": 0,
            "zx": int(zX), "zw": 0, "zy": int(zY),
            "depth_mult": 1,
            "input_slots": [slot_in],
            "input_ptrs": [in_ptr_0],
        })
        continue

    # ========== SOFTMAX COM QUANTIZAÇÃO (CORRIGIDO) ==========
    if optype == "SOFTMAX":
        if len(in_ids) < 1:
            continue

        in0 = subgraph.Tensors(in_ids[0])
        out0 = subgraph.Tensors(out_ids[0])

        in_shape = tensor_shape_list(in0)
        out_shape = tensor_shape_list(out0)

        in_h = in_shape[1] if len(in_shape) >= 3 else 1
        in_w = in_shape[2] if len(in_shape) >= 3 else 1
        cin = in_shape[3] if len(in_shape) >= 4 else (in_shape[1] if len(in_shape) == 2 else 1)

        out_h = out_shape[1] if len(out_shape) >= 3 else 1
        out_w = out_shape[2] if len(out_shape) >= 3 else 1
        cout = out_shape[3] if len(out_shape) >= 4 else (out_shape[1] if len(out_shape) == 2 else 1)

        sX = scale_scalar(in0)
        zX = zp_scalar(in0)
        sY = scale_scalar(out0)
        zY = zp_scalar(out0)

        beta = 1.0
        integer_bits = 5

        # Calcular internal_scale para documentação
        internal_scale = 1.0 / (1 << integer_bits)

        # Escala usada no caminho quantizado do softmax (estilo TFLite/TFLM)
        real_multiplier = beta * sX * (1 << (31 - integer_bits))
        real_multiplier = real_multiplier / (1 << 31)
        input_beta_mul, input_beta_left_shift = quantize_multiplier(real_multiplier)

        diff_min = -128  # padrão int8

        # Usar tensor_to_slot
        if in_ids[0] not in tensor_to_slot:
            raise RuntimeError(f"SOFTMAX op_index={i}: input tensor sem slot mapeado (tensor_id={in_ids[0]})")
        slot_in = tensor_to_slot[in_ids[0]]
        in_ptr_0 = slot_bases[slot_in]


        layer_params.append({
            "op_index": i,
            "optype": optype,
            "op_type": op_type,
            "act": ACT_NONE,
            "flags": 0,
            "in_slot": slot_in,
            "out_slot": out_slot,
            "in_h": int(in_h), "in_w": int(in_w),
            "cin": int(cin), "cout": int(cout),

            "kh": int(input_beta_mul),
            "kw": int(input_beta_left_shift),
            "stride_h": int(diff_min),
            "stride_w": int(integer_bits),

            "dil_h": 1, "dil_w": 1,
            "pad_t": in_ptr_0,
            "pad_b": 0, "pad_l": 0, "pad_r": 0,
            "out_h": int(out_h), "out_w": int(out_w),
            "w_off": 0,
            "has_bias": False,
            "b_off": 0,
            "has_mulq6": False,
            "mul_off": 0,
            "q6_off": 0,
            "zx": int(zX), "zw": 0, "zy": int(zY),
            "depth_mult": 1,
            "input_slots": [slot_in],
            "input_ptrs": [in_ptr_0],
            "quant_params": {
                "sX": float(sX), "sY": float(sY),
                "zX": int(zX), "zY": int(zY),
                "beta": float(beta),
                "integer_bits": int(integer_bits),
                "internal_scale": float(internal_scale),
                "input_beta_mul": int(input_beta_mul),
                "input_beta_left_shift": int(input_beta_left_shift),
                "diff_min": int(diff_min),
            }
        })
        continue

    # ========== OPERAÇÕES COM PESOS (CONV, DW, FC) ==========
    if len(in_ids) < 2:
        continue

    in0 = subgraph.Tensors(in_ids[0])
    wts = subgraph.Tensors(in_ids[1])
    out0 = subgraph.Tensors(out_ids[0])
    bias_id = int(in_ids[2]) if len(in_ids) >= 3 else -1

    in_shape = tensor_shape_list(in0)
    out_shape = tensor_shape_list(out0)
    w_shape = tensor_shape_list(wts)

    in_h = in_shape[1] if len(in_shape) >= 3 else 1
    in_w = in_shape[2] if len(in_shape) >= 3 else 1
    cin = in_shape[3] if len(in_shape) >= 4 else (in_shape[1] if len(in_shape) == 2 else 1)

    out_h = out_shape[1] if len(out_shape) >= 3 else 1
    out_w = out_shape[2] if len(out_shape) >= 3 else 1

    stride_h, stride_w = 1, 1
    dil_h, dil_w = 1, 1
    pad_t = pad_b = pad_l = pad_r = 0
    act = ACT_NONE
    flags = 0
    depth_mult = 1

    if optype == "CONV_2D":
        cout = int(w_shape[0]) if len(w_shape) >= 1 else int(out_shape[3] if len(out_shape) >= 4 else 1)
        kh = int(w_shape[1]) if len(w_shape) >= 2 else 1
        kw = int(w_shape[2]) if len(w_shape) >= 3 else 1

        stride_h, stride_w, dil_h, dil_w, pad_kind, act = parse_conv2d_options(op)
        if pad_kind == 0:
            flags |= FLAG_PADDING_SAME

    elif optype == "DEPTHWISE_CONV_2D":
        kh = int(w_shape[1]) if len(w_shape) >= 2 else 1
        kw = int(w_shape[2]) if len(w_shape) >= 3 else 1
        cout = int(w_shape[3]) if len(w_shape) >= 4 else int(out_shape[3] if len(out_shape) >= 4 else 1)

        stride_h, stride_w, dil_h, dil_w, pad_kind, act, depth_mult = parse_dwconv2d_options(op)
        if pad_kind == 0:
            flags |= FLAG_PADDING_SAME

    else:  # FULLY_CONNECTED
        cout = int(w_shape[0]) if len(w_shape) >= 1 else int(out_shape[1] if len(out_shape) >= 2 else 1)
        kh, kw = 1, 1
        act = parse_fc_options(op)

    if act == ACT_RELU6:
        flags |= FLAG_HAS_Q6

    if (flags & FLAG_PADDING_SAME) != 0:
        pad_t, pad_b, pad_l, pad_r, out_h, out_w = same_padding(
            in_h, in_w, kh, kw, stride_h, stride_w, dil_h, dil_w
        )

    zx = zp_scalar(in0)
    zw = zp_scalar(wts)
    zy = zp_scalar(out0)

    w_off = weight_tensor_off.get(in_ids[1], 0)
    has_bias = (bias_id >= 0 and bias_id in bias_tensor_off)
    b_off = int(bias_tensor_off.get(bias_id, 0))

    has_mulq6 = (i in mul_q6_off)
    mul_off = int(mul_q6_off[i][0]) if has_mulq6 else 0
    shift_off = int(mul_q6_off[i][1]) if has_mulq6 else 0
    q6_off = int(mul_q6_off[i][2]) if has_mulq6 else 0


    layer_params.append({
        "op_index": i,
        "optype": optype,
        "op_type": op_type,
        "act": act,
        "flags": flags,
        "in_slot": in_slots[0] if len(in_slots) > 0 else 0,
        "out_slot": out_slot,
        "in_h": int(in_h), "in_w": int(in_w),
        "cin": int(cin), "cout": int(cout),
        "kh": int(kh), "kw": int(kw),
        "stride_h": int(stride_h), "stride_w": int(stride_w),
        "dil_h": int(dil_h), "dil_w": int(dil_w),
        "pad_t": int(pad_t), "pad_b": int(pad_b),
        "pad_l": int(pad_l), "pad_r": int(pad_r),
        "out_h": int(out_h), "out_w": int(out_w),
        "w_off": int(w_off),
        "has_bias": bool(has_bias),
        "b_off": int(b_off),
        "has_mulq6": bool(has_mulq6),
        "mul_off": int(mul_off),
        "shift_off": int(shift_off),
        "q6_off": int(q6_off),
        "zx": int(zx), "zw": int(zw), "zy": int(zy),
        "depth_mult": int(depth_mult) if optype == "DEPTHWISE_CONV_2D" else 1,
        "input_slots": in_slots,
    })

# ============================================================
# (F) params blob + slots COM ENDEREÇOS CORRETOS
# ============================================================

def op_type_name(x):
    return {OP_CONV:"CONV", OP_DW:"DW", OP_FC:"FC", OP_ADD:"ADD", OP_MEAN:"MEAN", OP_SOFTMAX:"SOFTMAX", OP_QUANTIZE:"QUANTIZE"}.get(x, str(x))

def act_name(x):
    return {ACT_NONE:"NONE", ACT_RELU:"RELU", ACT_RELU6:"RELU6"}.get(x, str(x))

def flags_pretty(flags: int):
    parts = []
    if flags & FLAG_PADDING_SAME: parts.append("PADDING_SAME")
    if flags & FLAG_HAS_Q6: parts.append("HAS_Q6")
    return "|".join(parts) if parts else "0"

params_blob = bytearray()
layer_meta_full = []

# ===== DEBUG: Verificar slot_bases =====
print("\n" + "=" * 80)
print("DEBUG: SLOT_BASES:")
print("=" * 80)
print(f"slot_bases = {slot_bases}")
print(f"slot0_base = {slot0_base}")
print(f"SLOT_BYTES = {SLOT_BYTES}")
print()

# ===== VALIDAÇÃO FINAL (antes de serializar params_blob) =====

for li, p in enumerate(layer_params):  # ← USAR li ao invés de p["op_index"]
    if p["optype"] == "ADD":
        print(f"\nADD L{li} (op_index={p['op_index']}):")
        print(f"  input_slots: {p.get('input_slots', [])}")
        print(f"  pad_t (deve ser slot_bases[{p.get('input_slots', [None, None])[0]}]): {p['pad_t']}")
        print(f"  pad_b (deve ser slot_bases[{p.get('input_slots', [None, None])[1]}]): {p['pad_b']}")
        print(f"  slot_bases[{p.get('input_slots', [None, None])[0]}] = {slot_bases[p.get('input_slots', [None, None])[0]] if p.get('input_slots', [None, None])[0] is not None else 'N/A'}")
        print(f"  slot_bases[{p.get('input_slots', [None, None])[1]}] = {slot_bases[p.get('input_slots', [None, None])[1]] if len(p.get('input_slots', [])) > 1 else 'N/A'}")
        # pad_t/pad_b precisam ser bases de slot
        assert p["pad_t"] in slot_bases and p["pad_b"] in slot_bases, \
            f"ADD L{li} (op_index={p['op_index']}) com pad_t/pad_b fora de slot_bases"  # ← USAR li

        # in_slot deve bater com primeiro input slot
        if "input_slots" in p and len(p["input_slots"]) == 2:
            assert p["in_slot"] == p["input_slots"][0], \
                f"ADD L{li} (op_index={p['op_index']}) in_slot != input_slots[0]"  # ← USAR li
            assert p["pad_t"] == slot_bases[p["input_slots"][0]], \
                f"ADD L{li} (op_index={p['op_index']}) pad_t != base(input_slots[0])"  # ← USAR li
            assert p["pad_b"] == slot_bases[p["input_slots"][1]], \
                f"ADD L{li} (op_index={p['op_index']}) pad_b != base(input_slots[1])"  # ← USAR li

        if len(p.get("input_slots", [])) != 2:
            raise RuntimeError(f"ADD L{li} (op_index={p['op_index']}) sem 2 input_slots")  # ← USAR li

        # NOVO: ptrs dos dois inputs de ADD devem apontar para bases válidas de slot
        a, b = p["pad_t"], p["pad_b"]
        if a not in slot_bases or b not in slot_bases:
            raise RuntimeError(
                f"ADD L{li} (op_index={p['op_index']}) com ptr fora de slot_bases: {a}, {b}"  # ← USAR li
            )

        if p["pad_t"] == 0 or p["pad_b"] == 0:
            print(f"[WARN] ADD L{li} (op_index={p['op_index']}) com pad_t/pad_b zero")  # ← USAR li


for li, p in enumerate(layer_params):
    if p["optype"] == "ADD":
        # ADD usa dois ponteiros em pad_t/pad_b; no header in_ptr = primeiro input real
        in_ptr = p["pad_t"]
    else:
        in_ptr = slot_bases[p["in_slot"]]
    out_ptr = slot_bases[p["out_slot"]]

    wptr = kernel_base + p["w_off"]
    bias_ptr = (bias_base + p["b_off"]) if p["has_bias"] else 0
    mul_ptr = (mul_base + p["mul_off"]) if p["has_mulq6"] else 0
    shift_ptr = (shift_base + p["shift_off"]) if p["has_mulq6"] else 0
    q6_ptr = (q6_base + p["q6_off"]) if (p["has_mulq6"] and p["act"] == ACT_RELU6) else 0


    layer_meta_full.append(f";; ================== L{li} ==================")
    layer_meta_full.append(f";; op_index          : {p['op_index']}")
    layer_meta_full.append(f";; optype            : {p['optype']}")
    layer_meta_full.append(f";; op_type           : {p['op_type']} ({op_type_name(p['op_type'])})")
    layer_meta_full.append(f";; act               : {p['act']} ({act_name(p['act'])})")
    layer_meta_full.append(f";; flags             : {p['flags']} ({flags_pretty(p['flags'])})")

    # Mostrar slots de entrada
    if 'input_slots' in p and len(p['input_slots']) > 1:
        in_slots_str = ' e '.join(str(s) for s in p['input_slots'])
        layer_meta_full.append(f";; in_slot/out_slot  : [{in_slots_str}] -> {p['out_slot']}")
    else:
        layer_meta_full.append(f";; in_slot/out_slot  : {p['in_slot']} -> {p['out_slot']}")

    layer_meta_full.append(f";; in_ptr/out_ptr    : {in_ptr} -> {out_ptr}")

    # Para ADD: mostrar parâmetros de quantização
    if p['optype'] == 'ADD' and 'quant_params' in p:
        qp = p['quant_params']
        layer_meta_full.append(f";; --- ADD Quantization ---")
        layer_meta_full.append(f";; sA/sB/sY          : {qp['sA']:.6f} / {qp['sB']:.6f} / {qp['sY']:.6f}")
        layer_meta_full.append(f";; zA/zB/zY          : {qp['zA']} / {qp['zB']} / {qp['zY']}")
        layer_meta_full.append(f";; s_common          : {qp['s_common']:.6f}")
        layer_meta_full.append(f";; mul0/shift0       : {qp['mul0']} / {qp['shift0']}")
        layer_meta_full.append(f";; mul1/shift1       : {qp['mul1']} / {qp['shift1']}")
        layer_meta_full.append(f";; out_mul/out_shift : {qp['out_mul']} / {qp['out_shift']}")
        layer_meta_full.append(f";; input_ptrs        : {p['input_ptrs']}")

    # Para SOFTMAX: mostrar parâmetros de quantização
    if p['optype'] == 'SOFTMAX' and 'quant_params' in p:
        qp = p['quant_params']
        layer_meta_full.append(f";; --- SOFTMAX Quantization ---")
        layer_meta_full.append(f";; sX/sY              : {qp['sX']:.6f} / {qp['sY']:.6f}")
        layer_meta_full.append(f";; zX/zY              : {qp['zX']} / {qp['zY']}")
        layer_meta_full.append(f";; beta               : {qp['beta']:.6f}")
        layer_meta_full.append(f";; integer_bits       : {qp['integer_bits']}")
        layer_meta_full.append(f";; internal_scale     : {qp['internal_scale']:.6f}")
        layer_meta_full.append(f";; input_beta_mul     : {qp['input_beta_mul']}")
        layer_meta_full.append(f";; input_beta_left_sh : {qp['input_beta_left_shift']}")
        layer_meta_full.append(f";; diff_min           : {qp['diff_min']}")
    
    # Para QUANTIZE: mostrar parâmetros de quantização
    if p['optype'] == 'QUANTIZE' and 'quant_params' in p:
        qp = p['quant_params']
        layer_meta_full.append(f";; --- QUANTIZE Params ---")
        layer_meta_full.append(f";; scale_in/out      : {qp['scale_in']:.6f} / {qp['scale_out']:.6f}")
        layer_meta_full.append(f";; zp_in/out         : {qp['zp_in']} / {qp['zp_out']}")
        layer_meta_full.append(f";; ratio             : {qp['ratio']:.6f}")
        layer_meta_full.append(f";; mul/shift         : {qp['mul']} / {qp['shift']}")

    # Para operações multi-input, mostrar todos os ponteiros
    if 'input_ptrs' in p and len(p.get('input_slots', [])) > 1 and p['optype'] != 'ADD':
        layer_meta_full.append(f";; input_ptrs        : {p['input_ptrs'][:len(p['input_slots'])]}")

    layer_meta_full.append(f";; in_h/in_w         : {p['in_h']} x {p['in_w']}")
    layer_meta_full.append(f";; cin/cout          : {p['cin']} -> {p['cout']}")

    # Para ADD: mostrar mapeamento dos campos
    if p['optype'] == 'ADD':
        layer_meta_full.append(f";; kh/kw (mul0/shft0): {p['kh']} / {p['kw']}")
        layer_meta_full.append(f";; stride (mul1/sh1) : {p['stride_h']} / {p['stride_w']}")
        layer_meta_full.append(f";; dil (outM/outSh)  : {p['dil_h']} / {p['dil_w']}")
        layer_meta_full.append(f";; pad_t/b (inPtr)   : {p['pad_t']} / {p['pad_b']}")
        layer_meta_full.append(f";; pad_l/r (zA/zB)   : {p['pad_l']} / {p['pad_r']}")
    elif p['optype'] == 'MEAN':
        layer_meta_full.append(f";; kh/kw (mul/shift) : {p['kh']} / {p['kw']}")
        layer_meta_full.append(f";; stride_h (spatial): {p['stride_h']}")
        layer_meta_full.append(f";; pad_t (inPtr)     : {p['pad_t']}")
    elif p['optype'] == 'SOFTMAX':
        layer_meta_full.append(f";; kh/kw (βmul/shft) : {p['kh']} / {p['kw']}")
        layer_meta_full.append(f";; stride_h (diffMin): {p['stride_h']}")
        layer_meta_full.append(f";; stride_w (intBits): {p['stride_w']}")
        layer_meta_full.append(f";; pad_t (inPtr)     : {p['pad_t']}")
    else:
        layer_meta_full.append(f";; kh/kw             : {p['kh']} x {p['kw']}")
        layer_meta_full.append(f";; stride_h/stride_w : {p['stride_h']} x {p['stride_w']}")
        layer_meta_full.append(f";; dil_h/dil_w       : {p['dil_h']} x {p['dil_w']}")
        layer_meta_full.append(f";; pad t/b/l/r       : {p['pad_t']} {p['pad_b']} {p['pad_l']} {p['pad_r']}")

    layer_meta_full.append(f";; out_h/out_w       : {p['out_h']} x {p['out_w']}")
    if p["optype"] == "DEPTHWISE_CONV_2D":
        layer_meta_full.append(f";; depth_mult        : {p['depth_mult']}")
    layer_meta_full.append(f";; w_off/b_off       : {p['w_off']} / {p['b_off']}")
    layer_meta_full.append(f";; mul_off/q6_off     : {p['mul_off']} / {p['q6_off']}")
    layer_meta_full.append(f";; shift_off           : {p.get('shift_off', 0)}")
    layer_meta_full.append(f";; shift_ptr           : {shift_ptr}")

    layer_meta_full.append(f";; wptr/bias/mul/q6   : {wptr} / {bias_ptr} / {mul_ptr} / {q6_ptr}")
    layer_meta_full.append(f";; zx/zw/zy           : {p['zx']} / {p['zw']} / {p['zy']}")
    layer_meta_full.append(";;")

    params_blob += pack_layerparam(
        p["op_type"], p["act"], p["flags"],
        in_ptr, out_ptr,
        p["in_h"], p["in_w"], p["cin"], p["cout"],
        p["kh"], p["kw"],
        p["stride_h"], p["stride_w"],
        p["dil_h"], p["dil_w"],
        p["pad_t"], p["pad_b"], p["pad_l"], p["pad_r"],
        wptr, bias_ptr, mul_ptr, shift_ptr, q6_ptr,
        p["zx"], p["zw"], p["zy"],
        p["out_h"], p["out_w"]
    )

if len(params_blob) < params_bytes:
    params_blob += b"\x00" * (params_bytes - len(params_blob))

PAGE = 65536

def mem_pages_for(end_addr: int) -> int:
    return (end_addr + PAGE - 1) // PAGE

# ============================================================
# (G) build WAT
# ============================================================
sections = []
data_sections = []

sections.append(";; ===== AUTO-GERADO: weights/bias RAW + MUL/Q6 + params + slots =====")
sections.append(f";; MODEL: {MODEL_PATH}")
sections.append(f";; LayerParam size: {LP_SIZE} bytes | layers: {len(layer_params)}")
sections.append("")
sections.append(";; MAPEAMENTO DE CAMPOS PARA OPERAÇÕES ESPECIAIS:")
sections.append(";; ")
sections.append(";; CONVENÇÃO DE SHIFT (todas as operações):")
sections.append(";;   shift > 0: LEFT SHIFT (multiplicação por 2^shift)")
sections.append(";;   shift < 0: RIGHT SHIFT (divisão por 2^(-shift))")
sections.append(";; ")
sections.append(";; ADD (op_type=4):")
sections.append(";;   kh          = mul0 (Q31 multiplier para input 0)")
sections.append(";;   kw          = shift0 (pode ser negativo)")
sections.append(";;   stride_h    = mul1 (Q31 multiplier para input 1)")
sections.append(";;   stride_w    = shift1 (pode ser negativo)")
sections.append(";;   dil_h       = out_mul (Q31 multiplier para output)")
sections.append(";;   dil_w       = out_shift (pode ser negativo)")
sections.append(";;   pad_t       = input_ptr[0]")
sections.append(";;   pad_b       = input_ptr[1]")
sections.append(";;   pad_l       = zA (zero point input 0)")
sections.append(";;   pad_r       = zB (zero point input 1)")
sections.append(";;   zx          = zA")
sections.append(";;   zw          = zB")
sections.append(";;   zy          = zY")
sections.append(";; ")
sections.append(";; MEAN (op_type=5):")
sections.append(";;   kh          = mul (Q31 multiplier)")
sections.append(";;   kw          = shift (pode ser negativo)")
sections.append(";;   stride_h    = spatial_size (in_h * in_w)")
sections.append(";;   pad_t       = input_ptr")
sections.append(";; ")
sections.append(";; SOFTMAX (op_type=6):")
sections.append(";;   kh          = input_beta_mul (Q31 multiplier para input*beta)")
sections.append(";;   kw          = input_beta_left_shift (pode ser negativo)")
sections.append(";;   stride_h    = diff_min (limite inferior, tipicamente -128)")
sections.append(";;   stride_w    = integer_bits (para scaling interno, tipicamente 5)")
sections.append(";;   pad_t       = input_ptr")
sections.append(";;   zx          = zX (zero point input)")
sections.append(";;   zy          = zY (zero point output, tipicamente -128)")
sections.append("")
sections.append(";; QUANTIZE (op_type=7):")
sections.append(";;   kh          = mul (Q31 multiplier para conversão)")
sections.append(";;   kw          = shift (pode ser negativo)")
sections.append(";;   pad_t       = input_ptr")
sections.append(";;   zx          = zero_point_in")
sections.append(";;   zy          = zero_point_out")
sections.append(";;   Fórmula: out = ((in - zx) * mul >> shift) + zy")
sections.append(";; ")

sections.append(";; --- BASES (no overlap) ---")
sections.append(f"(global $PARAMS_BASE i32 (i32.const {params_base}))")
sections.append(f"(global $LP_SIZE i32 (i32.const {LP_SIZE}))")
sections.append(f"(global $SLOT0_BASE i32 (i32.const {slot_bases[0]}))")
sections.append(f"(global $NUM_LAYERS i32 (i32.const {len(layer_params)}))")
sections.append("")

sections.append(";; --- layer list (ALL ops) — FULL DUMP ---")
sections.extend(layer_meta_full)
sections.append("")

#data_sections.append(";; --- LUT (exp) (f32 little-endian) ---")
#data_sections.append(f";; bytes: {len(lut_blob)} @ base {LUT_BASE}")
#data_sections.append(wat_data_from_bytes(lut_blob, LUT_BASE))
#data_sections.append("")

data_sections.append(";; --- WEIGHTS (raw bytes) ---")
data_sections.append(f";; bytes: {kernel_bytes} @ base {kernel_base}")
data_sections.append(wat_data_from_bytes(bytes(weights_raw), kernel_base))
data_sections.append("")

data_sections.append(";; --- BIAS (raw bytes) ---")
data_sections.append(f";; bytes: {bias_bytes} @ base {bias_base}")
data_sections.append(wat_data_from_bytes(bytes(bias_raw), bias_base))
data_sections.append("")

data_sections.append(";; --- MUL table (int32) ---")
data_sections.append(f";; bytes: {mul_bytes} @ base {mul_base}")
data_sections.append(wat_data_from_bytes(mul_blob, mul_base))
data_sections.append("")

data_sections.append(";; --- SHIFT table (int32) ---")
data_sections.append(f";; bytes: {shift_bytes} @ base {shift_base}")
data_sections.append(wat_data_from_bytes(shift_blob, shift_base))
data_sections.append("")

data_sections.append(";; --- Q6 table (int32) ---")
data_sections.append(f";; bytes: {q6_bytes} @ base {q6_base}")
data_sections.append(wat_data_from_bytes(q6_blob, q6_base))
data_sections.append("")

data_sections.append(";; --- PARAMS: LayerParam array ---")
data_sections.append(f";; bytes: {len(params_blob)} @ base {params_base}")
data_sections.append(wat_data_from_bytes(bytes(params_blob), params_base))
data_sections.append("")

mem_end = max(
    #LUT_BASE + LUT_BYTES,
    kernel_base + kernel_bytes,
    bias_base + bias_bytes,
    mul_base + mul_bytes,
    q6_base + q6_bytes,
    params_base + len(params_blob),
    max(slot_bases) + SLOT_BYTES
)
mem_pages = mem_pages_for(mem_end)

wat_lines = []
wat_lines.append("(module")
wat_lines.append(f"  (memory (export \"memory\") {mem_pages})")
wat_lines.append("")

def indent_block(lines, n=2):
    pad = " " * n
    return [pad + ln if ln.strip() else "" for ln in lines]

wat_lines.extend(indent_block(sections, 2))
wat_lines.extend(indent_block(data_sections, 2))

wat_lines.append(")")

Path(OUT_WAT_PATH).write_text("\n".join(wat_lines), encoding="utf-8")

print("\n✅ OK:", OUT_WAT_PATH)
print(f"weights: base={kernel_base} bytes={kernel_bytes}")
print(f"bias:    base={bias_base} bytes={bias_bytes}")
print(f"mul:     base={mul_base} bytes={mul_bytes} entries={len(mul_vals)}")
print(f"q6:      base={q6_base} bytes={q6_bytes} entries={len(q6_vals)}")
print(f"params:  base={params_base} bytes={len(params_blob)} layers={len(layer_params)} LP_SIZE={LP_SIZE}")
print(f"slots:   SLOT_BYTES={SLOT_BYTES} bases={slot_bases}")

print("mul_base:", mul_base)
print("mul_bytes:", mul_bytes)
print("shift_base:", shift_base)
print("shift_bytes:", shift_bytes)
print("q6_base:", q6_base)
print("q6_bytes:", q6_bytes)

print("FC mul_off:", mul_off)
print("FC shift_off:", shift_off)
print("FC final mul_ptr:", mul_base + mul_off)
print("FC final shift_ptr:", shift_base + shift_off)

# Sumário de quantização para debug
print("\n" + "=" * 80)
print("RESUMO DE QUANTIZAÇÃO:")
print("=" * 80)
for li, p in enumerate(layer_params):
    if p['optype'] == 'ADD' and 'quant_params' in p:
        qp = p['quant_params']
        print(f"L{li} (ADD):")
        print(f"  sA={qp['sA']:.6f}, zA={qp['zA']}")
        print(f"  sB={qp['sB']:.6f}, zB={qp['zB']}")
        print(f"  sY={qp['sY']:.6f}, zY={qp['zY']}")
        print(f"  s_common={qp['s_common']:.6f}")
        print(f"  mul0={qp['mul0']}, shift0={qp['shift0']}")
        print(f"  mul1={qp['mul1']}, shift1={qp['shift1']}")
        print(f"  out_mul={qp['out_mul']}, out_shift={qp['out_shift']}")
    elif p['optype'] == 'SOFTMAX' and 'quant_params' in p:
        qp = p['quant_params']
        print(f"L{li} (SOFTMAX):")
        print(f"  sX={qp['sX']:.6f}, zX={qp['zX']}")
        print(f"  sY={qp['sY']:.6f}, zY={qp['zY']}")
        print(f"  beta={qp['beta']:.6f}")
        print(f"  integer_bits={qp['integer_bits']}, internal_scale={qp['internal_scale']:.6f}")
        print(f"  input_beta_mul={qp['input_beta_mul']}, input_beta_left_shift={qp['input_beta_left_shift']}")
        print(f"  diff_min={qp['diff_min']}")
    elif p['optype'] == 'QUANTIZE' and 'quant_params' in p:
        qp = p['quant_params']
        print(f"L{li} (QUANTIZE):")
        print(f"  scale_in={qp['scale_in']:.6f}, zp_in={qp['zp_in']}")
        print(f"  scale_out={qp['scale_out']:.6f}, zp_out={qp['zp_out']}")
        print(f"  ratio={qp['ratio']:.6f}")
        print(f"  mul={qp['mul']}, shift={qp['shift']}")
try:
    from google.colab import files
    files.download(OUT_WAT_PATH)
except Exception:
    pass