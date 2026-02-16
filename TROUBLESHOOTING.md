# 🔧 TROUBLESHOOTING - MobileNetV2 WASM

## 🚨 PROBLEMA DETECTADO

Sua rede está apresentando **saturação severa** na saída final, com 283 classes tendo o mesmo valor máximo (127). Isso indica um problema crítico na última camada.

## 📊 Análise do Resultado

```
❌ 283 classes com rawValue = 127 (saturação máxima)
❌ Todas têm a mesma probabilidade (0.35%)
❌ Entropia altíssima (8.4 bits) → distribuição quase uniforme
❌ Predição absurda: "komondor" (cachorro) para imagem de avião
```

## 🔍 Diagnóstico Passo a Passo

### PASSO 1: Execute o script de diagnóstico

```bash
node diagnose_network.js
```

Isso irá gerar:
- `diagnostic_report.json` - Relatório completo
- `output_values.txt` - Todos os 1000 valores de saída

### PASSO 2: Gere imagens de teste

```bash
node generate_test_images.js
```

Isso criará 9 imagens sintéticas para teste.

### PASSO 3: Teste com imagens simples

```bash
# Teste com imagem preta
cp test_black.raw aviao_uint8.raw
node test_mobilenetv2.js

# Teste com imagem branca
cp test_white.raw aviao_uint8.raw
node test_mobilenetv2.js
```

**O que verificar:**
- ✅ As saídas devem ser diferentes para black vs white
- ❌ Se forem iguais → problema nas camadas iniciais
- ❌ Se ambas saturarem → problema na quantização

## 🎯 Possíveis Causas e Soluções

### 1. Problema no SOFTMAX (Layer 66) - MAIS PROVÁVEL ⭐

**Sintomas:**
- Múltiplas classes com valor 127
- Distribuição uniforme

**Causa:**
- Parâmetros `input_beta_mul` e `input_beta_shift` incorretos
- `diff_min` muito restritivo
- Zero points (zX, zY) errados

**Solução:**
```wasm
;; Verificar na LayerParam do layer 66:
;; kh (9) = input_beta_mul   (deve ser Q31, ~1073741824 para beta=1.0)
;; kw (10) = input_beta_shift (geralmente -1 a -5)
;; stride_h (11) = diff_min   (geralmente -128 ou próximo)
;; stride_w (12) = integer_bits (geralmente 5)
;; pad_l (17) = zX (zero point entrada, geralmente próximo de 0)
;; zy (25) = zY (zero point saída, geralmente próximo de 0)
```

**Como corrigir:**
1. Verifique os parâmetros de quantização do Softmax no arquivo de pesos
2. Compare com a implementação de referência do TFLite
3. Ajuste `input_beta_mul` e `input_beta_shift` para não causar overflow

### 2. Problema no FULLY CONNECTED (Layer 65)

**Sintomas:**
- Valores já saturados antes do Softmax
- Todos os logits próximos de 127

**Causa:**
- Multiplicadores Q31 incorretos
- Bias muito altos
- Zero points errados

**Solução:**
```wasm
;; Verificar na LayerParam do layer 65:
;; cin (7) = 1280 (features da Mean)
;; cout (8) = 1000 (classes do ImageNet)
;; wptr (19) = endereço dos pesos (1280×1000 bytes)
;; bias_ptr (20) = endereço dos bias (1000 × 4 bytes = 4000 bytes)
;; mul_ptr (21) = multiplicadores Q31 (1000 × 4 bytes = 4000 bytes)
;; zx (23) = zero point entrada
;; zw (24) = zero point pesos (geralmente 0)
;; zy (25) = zero point saída
```

**Como verificar:**
1. Extraia os multiplicadores Q31 do arquivo de pesos
2. Para cada classe, o multiplicador deve ser: `M = (S_in * S_w) / S_out * 2^31`
3. Verifique se estão na ordem de 10^8 a 10^9

### 3. Problema no MEAN (Layer 64)

**Sintomas:**
- Features médias saturadas
- Todas próximas de 127 ou -128

**Causa:**
- Multiplicador de requantização errado
- Divisão por spatial_size incorreta

**Solução:**
```wasm
;; Verificar na LayerParam do layer 64:
;; in_h (5) = 7
;; in_w (6) = 7
;; cin (7) = 1280
;; kh (9) = mul (Q31) para requantização
;; kw (10) = shift
;; zx (23) = zero point entrada
;; zy (25) = zero point saída
```

## 🧪 Testes de Validação

### Teste 1: Valores intermediários

Modifique o WASM para exportar valores intermediários:

```wasm
(func $debug_layer (export "debug_layer") (param $layer_idx i32) (result i32)
  ;; Retorna estatísticas da camada
  ;; min, max, avg da saída
)
```

### Teste 2: Comparação com TFLite

Execute o mesmo modelo em TFLite/PyTorch e compare:
1. Logits antes do Softmax
2. Probabilidades finais
3. Valores de cada camada

### Teste 3: Gradiente de teste

Use o `test_gradient.raw` gerado e veja se a rede responde:
- Se todas as posições do gradiente produzem a mesma saída → problema
- Se há variação → rede está funcionando, mas pesos errados

## 📋 Checklist de Verificação

### Layer 65 (Fully Connected)
- [ ] Pesos: 1280 × 1000 = 1,280,000 bytes
- [ ] Bias: 1000 × 4 = 4,000 bytes (int32)
- [ ] Multiplicadores: 1000 × 4 = 4,000 bytes (int32 Q31)
- [ ] Zero points: zx, zw, zy estão corretos?
- [ ] Endereços: wptr, bias_ptr, mul_ptr são válidos?

### Layer 66 (Softmax)
- [ ] input_beta_mul ≈ 1073741824 para beta=1.0?
- [ ] input_beta_shift entre -5 e 0?
- [ ] diff_min = -128 ou próximo?
- [ ] integer_bits = 5?
- [ ] Zero points: zX ≈ 0, zY ≈ 0?

### Memória WASM
- [ ] Não há overlap entre buffers?
- [ ] Tamanho da memória é suficiente?
- [ ] Endereços estão alinhados corretamente?

## 🔨 Ferramentas de Debug

### 1. Extrair parâmetros do WASM

```bash
# Use wasm-objdump para ver os dados
wasm-objdump -x main.wasm | grep -A 20 "data"
```

### 2. Hexdump dos pesos

```bash
# Extrair região de memória específica
dd if=main.wasm of=weights_fc.bin bs=1 skip=<offset> count=1280000
hexdump -C weights_fc.bin | head -20
```

### 3. Validar multiplicadores Q31

```javascript
// Verificar se multiplicador Q31 está na faixa correta
const mul = 1234567890; // exemplo
const as_q31 = mul / Math.pow(2, 31);
console.log(`Q31: ${mul} = ${as_q31} (deve estar entre 0.5 e 2.0)`);
```

## 🎓 Referências Úteis

1. **TFLite Quantization**: https://www.tensorflow.org/lite/performance/quantization_spec
2. **Q31 Format**: Fixed-point Q31 representa [-1, 1) com 31 bits fracionários
3. **Softmax Int8**: Usa tabela de lookup ou aproximação polinomial
4. **Zero Points**: Representam o valor "0" real no espaço quantizado

## 💡 Dica Final

Se você tem acesso ao modelo original (TFLite ou ONNX):
1. Use `netron` para visualizar a arquitetura
2. Compare os parâmetros de quantização
3. Verifique a ordem das camadas
4. Confirme os tamanhos dos tensores

Se o problema persistir após verificar tudo isso, o mais provável é que:
- Os **multiplicadores Q31 da FC** estão muito altos (causando overflow)
- O **Softmax não está implementado corretamente** para int8
- Os **pesos foram corrompidos** durante a conversão

Boa sorte! 🍀
