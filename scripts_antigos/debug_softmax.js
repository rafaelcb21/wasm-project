const fs = require('fs');

function rgb888ToRgb565(rgb888Buffer, width, height) {
    const rgb565Buffer = new Uint16Array(width * height);
    for (let i = 0; i < width * height; i++) {
        const r = rgb888Buffer[i * 3 + 0];
        const g = rgb888Buffer[i * 3 + 1];
        const b = rgb888Buffer[i * 3 + 2];
        const r5 = (r >> 3) & 0x1F;
        const g6 = (g >> 2) & 0x3F;
        const b5 = (b >> 3) & 0x1F;
        rgb565Buffer[i] = (b5 << 11) | (g6 << 5) | r5;
    }
    return rgb565Buffer;
}

(async () => {
    console.log('🔍 DEBUG SOFTMAX ESPECÍFICO\n');
    
    try {
        const wasmBuffer = fs.readFileSync("main.wasm");
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const instance = wasmModule.instance;
        const memory = instance.exports.memory;
        
        // Carregar imagem
        const rawImage = fs.readFileSync("aviao_224x224x3.raw");
        const rgb565Image = rgb888ToRgb565(rawImage, 224, 224);
        const rgb565Bytes = new Uint8Array(rgb565Image.buffer);
        
        const inputPtr = 1767856;
        new Uint8Array(memory.buffer).set(rgb565Bytes, inputPtr);
        
        console.log('✅ Imagem carregada\n');
        console.log('═══════════════════════════════════════════════════════\n');
        
        // Executar layers até ANTES do softmax
        console.log('📊 Executando até Layer 63 (antes do Softmax)...\n');
        
        for (let i = 0; i <= 63; i++) {
            instance.exports.run_layer(i);
        }
        
        console.log('✅ Layers 0-63 executadas\n');
        
        // Ler parâmetros da layer 64 (Softmax)
        const paramsBase = 1760576 + 64 * 112;
        const params = new Int32Array(memory.buffer, paramsBase, 28);
        
        console.log('📋 PARÂMETROS DA LAYER 64 (Softmax):\n');
        console.log(`   op_type: ${params[0]} (esperado: 6 = SOFTMAX)`);
        console.log(`   in_ptr (pad_t): ${params[15]}`);
        console.log(`   out_ptr: ${params[4]}`);
        console.log(`   cin: ${params[7]}`);
        console.log(`   input_beta_mul (kh): ${params[9]}`);
        console.log(`   input_beta_shift (kw): ${params[10]}`);
        console.log(`   diff_min (stride_h): ${params[11]}`);
        console.log(`   zX: ${params[23]}`);
        console.log(`   zY: ${params[25]}\n`);
        
        // Ler INPUT do softmax (saída da FC)
        const softmaxInputPtr = params[15]; // pad_t = input_ptr
        const softmaxInputArray = new Int8Array(memory.buffer, softmaxInputPtr, 1000);
        
        console.log('📊 INPUT DO SOFTMAX (saída da FC - Layer 63):\n');
        
        const inputValues = Array.from(softmaxInputArray);
        const inputStats = {
            min: Math.min(...inputValues),
            max: Math.max(...inputValues),
            avg: inputValues.reduce((a, b) => a + b, 0) / inputValues.length,
            unique: new Set(inputValues).size
        };
        
        console.log(`   Min: ${inputStats.min}`);
        console.log(`   Max: ${inputStats.max}`);
        console.log(`   Avg: ${inputStats.avg.toFixed(2)}`);
        console.log(`   Valores únicos: ${inputStats.unique}/1000`);
        console.log(`   Sample [0-19]: [${inputValues.slice(0, 20).join(', ')}]`);
        
        // Histograma dos top valores
        const sorted = [...inputValues].sort((a, b) => b - a);
        console.log(`\n   Top 10 valores: [${sorted.slice(0, 10).join(', ')}]`);
        console.log(`   Bottom 10 valores: [${sorted.slice(-10).join(', ')}]\n`);
        
        // Verificar se input está válido
        if (inputStats.unique === 1) {
            console.log('❌ PROBLEMA: Input do Softmax tem apenas 1 valor único!');
            console.log('   A FC (Layer 63) não está funcionando corretamente.\n');
            return;
        }
        
        console.log('✅ Input do Softmax parece válido\n');
        console.log('═══════════════════════════════════════════════════════\n');
        
        // Agora executar o Softmax
        console.log('🔄 Executando Layer 64 (Softmax)...\n');
        instance.exports.run_layer(64);
        
        // Ler OUTPUT do softmax
        const softmaxOutputPtr = params[4];
        const softmaxOutputArray = new Int8Array(memory.buffer, softmaxOutputPtr, 1000);
        
        console.log('📊 OUTPUT DO SOFTMAX:\n');
        
        const outputValues = Array.from(softmaxOutputArray);
        const outputStats = {
            min: Math.min(...outputValues),
            max: Math.max(...outputValues),
            avg: outputValues.reduce((a, b) => a + b, 0) / outputValues.length,
            unique: new Set(outputValues).size
        };
        
        console.log(`   Min: ${outputStats.min}`);
        console.log(`   Max: ${outputStats.max}`);
        console.log(`   Avg: ${outputStats.avg.toFixed(2)}`);
        console.log(`   Valores únicos: ${outputStats.unique}/1000`);
        console.log(`   Sample [0-19]: [${outputValues.slice(0, 20).join(', ')}]`);
        
        // Comparar input vs output
        console.log('\n🔍 COMPARAÇÃO INPUT → OUTPUT:\n');
        console.log('   Índice | Input | Output | Diff');
        console.log('   -------|-------|--------|------');
        
        for (let i = 0; i < 20; i++) {
            const diff = outputValues[i] - inputValues[i];
            console.log(`   ${i.toString().padStart(6)} | ${inputValues[i].toString().padStart(5)} | ${outputValues[i].toString().padStart(6)} | ${diff.toString().padStart(4)}`);
        }
        
        console.log('\n═══════════════════════════════════════════════════════\n');
        
        // Diagnóstico
        if (outputStats.unique === 1) {
            console.log('❌ PROBLEMA CONFIRMADO: Softmax está colapsando tudo!\n');
            console.log('💡 POSSÍVEIS CAUSAS:\n');
            console.log('   1. diff_min muito restritivo (clamping excessivo)');
            console.log('   2. input_beta_mul/shift incorretos');
            console.log('   3. zX/zY causando saturação');
            console.log('   4. Overflow na multiplicação quantizada\n');
            
            // Testar manualmente a lógica do softmax
            console.log('🧪 TESTE MANUAL DA LÓGICA:\n');
            
            const testVal = inputValues[0];
            console.log(`   Valor de teste: ${testVal}`);
            console.log(`   testVal - zX = ${testVal - params[23]}`);
            
            // Simular multiply_by_quantized_multiplier
            const afterSub = testVal - params[23];
            console.log(`   Após subtração zX: ${afterSub}`);
            console.log(`   input_beta_mul: ${params[9]}`);
            console.log(`   input_beta_shift: ${params[10]}`);
            
            // O problema provavelmente está aqui
            console.log(`   diff_min (clamp): ${params[11]}`);
            console.log(`   zY (offset final): ${params[25]}`);
            
        } else {
            console.log('✅ Softmax produziu valores variados!\n');
            
            // Mostrar top-10
            const ranked = outputValues
                .map((val, idx) => ({ idx, val }))
                .sort((a, b) => b.val - a.val);
            
            console.log('🏆 TOP-10 CLASSES (por valor raw):\n');
            for (let i = 0; i < 10; i++) {
                console.log(`   ${i + 1}. Classe ${ranked[i].idx}: ${ranked[i].val}`);
            }
        }
        
        // Salvar para análise
        const debugData = {
            softmaxParams: {
                input_beta_mul: params[9],
                input_beta_shift: params[10],
                diff_min: params[11],
                zX: params[23],
                zY: params[25]
            },
            input: {
                stats: inputStats,
                values: inputValues.slice(0, 100)
            },
            output: {
                stats: outputStats,
                values: outputValues.slice(0, 100)
            }
        };
        
        fs.writeFileSync('debug_softmax.json', JSON.stringify(debugData, null, 2));
        console.log('\n💾 Debug salvo em debug_softmax.json\n');
        
    } catch (error) {
        console.error('\n❌ ERRO:', error.message);
        console.error(error.stack);
    }
})();