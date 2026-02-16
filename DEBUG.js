const fs = require("fs");

const IMAGE_PATH = "aviao_uint8.raw";
const WASM_PATH = "main.wasm";

const INPUT_PTR = 1800512;
const PARAMS_BASE = 1792736;
const LP_SIZE = 116;
const NUM_LAYERS = 67;
const NUM_CLASSES = 1000;

const FIELD = {
    op_type: 0,
    act: 1,
    flags: 2,
    in_ptr: 3,
    out_ptr: 4,
    in_h: 5,
    in_w: 6,
    cin: 7,
    cout: 8,
    kh: 9,
    kw: 10,
    stride_h: 11,
    stride_w: 12,
    wptr: 19,
    bias_ptr: 20,
    mul_ptr: 21,
    q6_ptr: 22,
    zx: 23,
    zw: 24,
    zy: 25,
    out_h: 26,
    out_w: 27
};

function isValidPtr(ptr, memorySize) {
    return ptr >= 0 && ptr < memorySize;
}

function analyzeLayer(memory, layerIdx) {
    const base = PARAMS_BASE + layerIdx * LP_SIZE;
    const view = new Int32Array(memory.buffer, base, 28);
    
    const opType = view[FIELD.op_type];
    const inPtr = view[FIELD.in_ptr];
    const outPtr = view[FIELD.out_ptr];
    const cin = view[FIELD.cin];
    const cout = view[FIELD.cout];
    const outH = view[FIELD.out_h];
    const outW = view[FIELD.out_w];
    
    const opNames = {
        1: "CONV", 2: "DW", 3: "FC", 4: "ADD", 
        5: "MEAN", 6: "SOFTMAX", 7: "QUANTIZE"
    };
    
    console.log(`\n📍 Layer ${layerIdx} (${opNames[opType] || opType})`);
    console.log(`   in_ptr: ${inPtr}, out_ptr: ${outPtr}`);
    console.log(`   cin: ${cin}, cout: ${cout}`);
    console.log(`   out_h: ${outH}, out_w: ${outW}`);
    
    const memSize = memory.buffer.byteLength;
    
    // Verificar validade dos ponteiros
    if (!isValidPtr(inPtr, memSize)) {
        console.log(`   ⚠️  AVISO: in_ptr inválido (${inPtr})`);
        return { min: 0, max: 0, unique: 0, zeros: 0 };
    }
    
    if (!isValidPtr(outPtr, memSize)) {
        console.log(`   ⚠️  AVISO: out_ptr inválido (${outPtr})`);
        return { min: 0, max: 0, unique: 0, zeros: 0 };
    }
    
    try {
        // Analisar entrada
        const inSize = Math.min(1000, cin * (outH || 1) * (outW || 1));
        
        if (inSize > 0 && inPtr + inSize <= memSize) {
            const inData = new Int8Array(memory.buffer, inPtr, inSize);
            const inStats = getStats(inData);
            console.log(`   📥 Input: min=${inStats.min}, max=${inStats.max}, unique=${inStats.unique}, zeros=${inStats.zeros}`);
        } else {
            console.log(`   📥 Input: SKIP (size=${inSize})`);
        }
        
        // Analisar saída
        const outSize = Math.min(1000, cout * (outH || 1) * (outW || 1));
        
        if (outSize > 0 && outPtr + outSize <= memSize) {
            const outData = new Int8Array(memory.buffer, outPtr, outSize);
            const outStats = getStats(outData);
            console.log(`   📤 Output: min=${outStats.min}, max=${outStats.max}, unique=${outStats.unique}, zeros=${outStats.zeros}`);
        } else {
            console.log(`   📤 Output: SKIP (size=${outSize})`);
        }
        
        // Analisar pesos se for Conv/DW/FC
        if ([1, 2, 3].includes(opType)) {
            const wptr = view[FIELD.wptr];
            const biasPtr = view[FIELD.bias_ptr];
            const mulPtr = view[FIELD.mul_ptr];
            
            if (wptr && isValidPtr(wptr, memSize)) {
                const wSize = Math.min(100, cin * cout);
                if (wptr + wSize <= memSize) {
                    const weights = new Int8Array(memory.buffer, wptr, wSize);
                    const wStats = getStats(weights);
                    console.log(`   ⚖️  Weights: min=${wStats.min}, max=${wStats.max}, unique=${wStats.unique}`);
                }
            }
            
            if (biasPtr && isValidPtr(biasPtr, memSize)) {
                const biasSize = Math.min(10, cout);
                if (biasPtr + biasSize * 4 <= memSize) {
                    const biases = new Int32Array(memory.buffer, biasPtr, biasSize);
                    console.log(`   📊 Bias sample: [${Array.from(biases.slice(0, 5)).join(', ')}]`);
                }
            }
            
            if (mulPtr && isValidPtr(mulPtr, memSize)) {
                const mulSize = Math.min(10, cout);
                if (mulPtr + mulSize * 4 <= memSize) {
                    const muls = new Int32Array(memory.buffer, mulPtr, mulSize);
                    console.log(`   🔢 Multiplier sample: [${Array.from(muls.slice(0, 5)).join(', ')}]`);
                }
            }
            
            console.log(`   🎯 zx=${view[FIELD.zx]}, zw=${view[FIELD.zw]}, zy=${view[FIELD.zy]}`);
        }
    } catch (e) {
        console.log(`   ❌ ERRO ao analisar layer: ${e.message}`);
    }
    
    return { min: 0, max: 0, unique: 0, zeros: 0 };
}

function getStats(arr) {
    const values = Array.from(arr);
    return {
        min: Math.min(...values),
        max: Math.max(...values),
        unique: new Set(values).size,
        zeros: values.filter(v => v === 0).length,
        mean: values.reduce((a, b) => a + b, 0) / values.length
    };
}

// Após carregar o WASM
function testFullyConnected(memory, instance) {
    console.log("\n🧪 TESTE ESPECÍFICO DA FULLY CONNECTED\n");
    
    const fcIdx = 64;
    const base = PARAMS_BASE + fcIdx * LP_SIZE;
    const view = new Int32Array(memory.buffer, base, 28);
    
    const inPtr = view[FIELD.in_ptr];
    const outPtr = view[FIELD.out_ptr];
    const cin = view[FIELD.cin];
    const cout = view[FIELD.cout];
    
    // Verificar entrada antes da FC
    const input = new Int8Array(memory.buffer, inPtr, cin);
    console.log("Input da FC:");
    console.log(`  Tamanho: ${cin}`);
    console.log(`  Min: ${Math.min(...input)}`);
    console.log(`  Max: ${Math.max(...input)}`);
    console.log(`  Valores únicos: ${new Set(input).size}`);
    console.log(`  Primeiros 10: [${Array.from(input.slice(0, 10)).join(', ')}]`);
    
    // Executar só a FC
    instance.exports.run_layer(fcIdx);
    
    // Verificar saída
    const output = new Int8Array(memory.buffer, outPtr, cout);
    console.log("\nOutput da FC:");
    console.log(`  Tamanho: ${cout}`);
    console.log(`  Min: ${Math.min(...output)}`);
    console.log(`  Max: ${Math.max(...output)}`);
    console.log(`  Valores únicos: ${new Set(output).size}`);
    console.log(`  Primeiros 20: [${Array.from(output.slice(0, 20)).join(', ')}]`);
    
    // Verificar pesos e bias
    const wptr = view[FIELD.wptr];
    const biasPtr = view[FIELD.bias_ptr];
    const weights = new Int8Array(memory.buffer, wptr, Math.min(1000, cin * cout));
    const biases = new Int32Array(memory.buffer, biasPtr, Math.min(10, cout));
    
    console.log("\nParâmetros:");
    console.log(`  Weight stats: min=${Math.min(...weights)}, max=${Math.max(...weights)}`);
    console.log(`  Bias sample: [${Array.from(biases.slice(0, 5)).join(', ')}]`);
}

(async () => {
    console.log("🚀 Executando MobileNetV2 com Debug\n");

    const wasmBuffer = fs.readFileSync(WASM_PATH);
    const { instance } = await WebAssembly.instantiate(wasmBuffer, {
        env: {
            log: (value) => console.log("LOG:", value),
        }
    });

    const memory = instance.exports.memory;
    const memU8 = new Uint8Array(memory.buffer);

    console.log("✅ WASM carregado");
    console.log(`💾 Memória: ${memory.buffer.byteLength} bytes`);

    // Inserir imagem
    const image = fs.readFileSync(IMAGE_PATH);
    
    if (image.length !== 224 * 224 * 3) {
        console.error("❌ Tamanho incorreto da imagem!");
        process.exit(1);
    }

    memU8.set(image, INPUT_PTR);
    
    // Verificar imagem carregada
    const imgStats = getStats(new Int8Array(memory.buffer, INPUT_PTR, 1000));
    console.log(`📷 Imagem inserida: min=${imgStats.min}, max=${imgStats.max}, unique=${imgStats.unique}\n`);

    // Executar camada por camada com debug
    console.log("⚙️ Executando camadas...\n");
    
    // Vamos executar TODAS as camadas e ver qual delas falha primeiro
    for (let i = 0; i < NUM_LAYERS; i++) {
        console.log(`\n${"=".repeat(60)}`);
        console.log(`🔄 EXECUTANDO LAYER ${i}`);
        console.log("=".repeat(60));
        
        // Mostrar estado ANTES da execução
        console.log("\n📋 ANTES:");
        analyzeLayer(memory, i);
        
        // Executar a camada
        try {
            instance.exports.run_layer(i);
            console.log(`\n✅ Layer ${i} executada com sucesso`);
        } catch (e) {
            console.log(`\n❌ ERRO ao executar layer ${i}: ${e.message}`);
            break;
        }
        
        // Mostrar estado DEPOIS da execução
        console.log("\n📋 DEPOIS:");
        analyzeLayer(memory, i);
        
        // Parar após algumas camadas críticas para não poluir output
        if (i >= 5) {
            console.log("\n⏭️  Pulando para camadas finais...");
            
            // Executar o resto sem debug
            for (let j = i + 1; j < 62; j++) {
                instance.exports.run_layer(j);
            }
            
            // Debug das últimas camadas
            for (let j = 62; j < NUM_LAYERS; j++) {
                console.log(`\n${"=".repeat(60)}`);
                console.log(`🔄 EXECUTANDO LAYER ${j}`);
                console.log("=".repeat(60));
                
                console.log("\n📋 ANTES:");
                analyzeLayer(memory, j);
                
                instance.exports.run_layer(j);
                
                console.log("\n📋 DEPOIS:");
                analyzeLayer(memory, j);
            }
            
            break;
        }
    }

    // Análise final
    console.log("\n" + "=".repeat(60));
    console.log("📊 RESULTADO FINAL");
    console.log("=".repeat(60));

    const resultPtr = instance.exports.get_result_ptr();
    console.log(`\nResult pointer: ${resultPtr}`);
    
    const finalOutput = new Int8Array(memory.buffer, resultPtr, NUM_CLASSES);
    const finalStats = getStats(finalOutput);

    console.log(`\nEstatísticas finais:`);
    console.log(`Min: ${finalStats.min}`);
    console.log(`Max: ${finalStats.max}`);
    console.log(`Valores únicos: ${finalStats.unique}`);
    console.log(`Zeros: ${finalStats.zeros}/${NUM_CLASSES}`);
    console.log(`Mean: ${finalStats.mean.toFixed(3)}`);
    console.log(`\nPrimeiros 50: [${Array.from(finalOutput.slice(0, 50)).join(', ')}]`);
    console.log(`\nÚltimos 50: [${Array.from(finalOutput.slice(-50)).join(', ')}]`);

    // Se houver variação, mostrar top-5
    if (finalStats.unique > 1) {
        const scale = 0.00390625;
        const logitsFloat = Array.from(finalOutput).map(v => v * scale);
        
        function softmax(arr) {
            const max = Math.max(...arr);
            const exps = arr.map(v => Math.exp(v - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(v => v / sum);
        }
        
        const probs = softmax(logitsFloat);
        const top5 = probs
            .map((p, i) => ({ index: i, prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5);

        console.log("\n🏆 TOP-5 RESULTADOS:\n");
        top5.forEach((r, i) => {
            console.log(`${i + 1}. ${(r.prob * 100).toFixed(2)}% → classe ${r.index}`);
        });
    } else {
        console.log("\n❌ ERRO: Todos os valores são idênticos!");
        
        // Verificar se é problema de zero points
        console.log("\n🔍 Investigando possível problema de zero-points...");
        
        // Verificar layer 64 (FC) especificamente
        const fcBase = PARAMS_BASE + 64 * LP_SIZE;
        const fcView = new Int32Array(memory.buffer, fcBase, 28);
        
        console.log("\n📊 Layer 64 (FC) - Parâmetros de quantização:");
        console.log(`   zx (input zp): ${fcView[FIELD.zx]}`);
        console.log(`   zw (weight zp): ${fcView[FIELD.zw]}`);
        console.log(`   zy (output zp): ${fcView[FIELD.zy]}`);
        
        const fcInPtr = fcView[FIELD.in_ptr];
        const fcOutPtr = fcView[FIELD.out_ptr];
        const fcCin = fcView[FIELD.cin];
        
        console.log(`\n   Input pointer: ${fcInPtr}`);
        console.log(`   Output pointer: ${fcOutPtr}`);
        console.log(`   Cin: ${fcCin}`);
        
        if (isValidPtr(fcInPtr, memory.buffer.byteLength)) {
            const fcInput = new Int8Array(memory.buffer, fcInPtr, Math.min(100, fcCin));
            const fcInputStats = getStats(fcInput);
            console.log(`\n   Input stats: min=${fcInputStats.min}, max=${fcInputStats.max}, unique=${fcInputStats.unique}`);
            console.log(`   Input sample: [${Array.from(fcInput.slice(0, 20)).join(', ')}]`);
        }
    }

    // Execute no final do código anterior
    testFullyConnected(memory, instance);

})();