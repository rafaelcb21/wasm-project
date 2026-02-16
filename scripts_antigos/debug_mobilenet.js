const fs = require('fs');

// Função para converter RGB888 para RGB565
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

// Função para imprimir estatísticas de um array int8
function printStats(arr, name) {
    const values = Array.from(arr);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const nonZero = values.filter(v => v !== 0).length;
    
    console.log(`\n📊 ${name}:`);
    console.log(`   Min: ${min}, Max: ${max}, Avg: ${avg.toFixed(2)}`);
    console.log(`   Non-zero: ${nonZero}/${values.length}`);
    console.log(`   Sample [0-9]: [${values.slice(0, 10).join(', ')}]`);
    
    // Histograma simples
    const hist = {};
    values.forEach(v => hist[v] = (hist[v] || 0) + 1);
    const sorted = Object.entries(hist).sort((a, b) => b[1] - a[1]).slice(0, 5);
    console.log(`   Top 5 valores: ${sorted.map(([v, c]) => `${v}(${c}x)`).join(', ')}`);
}

(async () => {
    console.log('🔍 DEBUG MODE - MobileNetV2\n');
    
    try {
        const wasmBuffer = fs.readFileSync("main.wasm");
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const instance = wasmModule.instance;
        const memory = instance.exports.memory;
        const memoryView = new Uint8Array(memory.buffer);
        
        console.log('✅ WASM carregado\n');
        
        // Carregar imagem
        const rawImage = fs.readFileSync("aviao_224x224x3.raw");
        const rgb565Image = rgb888ToRgb565(rawImage, 224, 224);
        const rgb565Bytes = new Uint8Array(rgb565Image.buffer);
        
        const inputPtr = 1767856;
        memoryView.set(rgb565Bytes, inputPtr);
        
        console.log('✅ Imagem carregada\n');
        
        // Executar camadas uma por uma e inspecionar
        console.log('═══════════════════════════════════════════════');
        console.log('  EXECUTANDO CAMADAS INDIVIDUALMENTE');
        console.log('═══════════════════════════════════════════════\n');
        
        const layersToDebug = [0, 1, 2, 10, 20, 30, 40, 50, 60, 61, 62, 63, 64];
        
        for (const layerIdx of layersToDebug) {
            console.log(`\n🔄 Executando Layer ${layerIdx}...`);
            
            try {
                instance.exports.run_layer(layerIdx);
                
                // Ler parâmetros da camada
                const paramsBase = 1760576 + layerIdx * 112;
                const params = new Int32Array(memory.buffer, paramsBase, 28);
                
                const opType = params[0];
                const outPtr = params[4];
                const outH = params[26];
                const outW = params[27];
                const cout = params[8];
                
                const opNames = ['', 'CONV', 'DW', 'FC', 'ADD', 'MEAN', 'SOFTMAX'];
                
                console.log(`   Tipo: ${opNames[opType] || opType}`);
                console.log(`   Output: ${outH}x${outW}x${cout} @ ${outPtr}`);
                
                // Ler saída
                const outputSize = (opType === 6) ? 1000 : // SOFTMAX
                                  (opType === 5) ? cout :  // MEAN
                                  (opType === 3) ? cout :  // FC
                                  outH * outW * cout;      // CONV/DW/ADD
                
                const output = new Int8Array(memory.buffer, outPtr, Math.min(outputSize, 1000));
                
                printStats(output, `Layer ${layerIdx} Output`);
                
                // Parar se todos os valores são -128 ou -127
                const allMin = Array.from(output).every(v => v <= -126);
                if (allMin && layerIdx > 0) {
                    console.log('\n⚠️  PROBLEMA DETECTADO: Todos os valores estão saturados!');
                    console.log('    A rede está produzindo apenas valores mínimos.');
                    break;
                }
                
            } catch (e) {
                console.error(`\n❌ ERRO na Layer ${layerIdx}:`, e.message);
                break;
            }
        }
        
        console.log('\n═══════════════════════════════════════════════');
        console.log('  ANÁLISE COMPLETA');
        console.log('═══════════════════════════════════════════════\n');
        
        // Verificar resultado final
        const resultPtr = instance.exports.get_result_ptr();
        const finalOutput = new Int8Array(memory.buffer, resultPtr, 1000);
        
        printStats(finalOutput, 'Saída Final (1000 classes)');
        
        // Analisar distribuição
        const uniqueValues = new Set(Array.from(finalOutput));
        console.log(`\n🔢 Valores únicos na saída: ${uniqueValues.size}`);
        
        if (uniqueValues.size === 1) {
            console.log('❌ PROBLEMA CRÍTICO: Todos os valores são iguais!');
        } else if (uniqueValues.size < 10) {
            console.log('⚠️  Muito poucos valores únicos - quantização pode estar muito agressiva');
        }
        
        // Salvar outputs intermediários
        console.log('\n💾 Salvando outputs para análise...');
        
        const debugData = {
            inputStats: {
                min: Math.min(...Array.from(rgb565Bytes)),
                max: Math.max(...Array.from(rgb565Bytes)),
                size: rgb565Bytes.length
            },
            finalOutput: Array.from(finalOutput).slice(0, 100), // Primeiras 100 classes
            uniqueValues: uniqueValues.size,
            allValuesAreSame: uniqueValues.size === 1
        };
        
        fs.writeFileSync('debug_output.json', JSON.stringify(debugData, null, 2));
        fs.writeFileSync('final_output.bin', Buffer.from(finalOutput));
        
        console.log('✅ debug_output.json salvo');
        console.log('✅ final_output.bin salvo\n');
        
    } catch (error) {
        console.error('\n❌ ERRO:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();