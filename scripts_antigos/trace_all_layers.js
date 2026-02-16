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
    console.log('🔍 RASTREAMENTO COMPLETO DE TODAS AS LAYERS\n');
    
    const wasmBuffer = fs.readFileSync("main.wasm");
    const wasmModule = await WebAssembly.instantiate(wasmBuffer);
    const instance = wasmModule.instance;
    const memory = instance.exports.memory;
    
    // Usar imagem branca (mais fácil de debugar)
    const whiteImage = Buffer.alloc(224 * 224 * 3, 255);
    const rgb565 = rgb888ToRgb565(whiteImage, 224, 224);
    const rgb565Bytes = new Uint8Array(rgb565.buffer);
    
    const inputPtr = 1767856;
    new Uint8Array(memory.buffer).set(rgb565Bytes, inputPtr);
    
    console.log('✅ Imagem BRANCA carregada\n');
    console.log('═══════════════════════════════════════════════════════\n');
    
    const layerReport = [];
    
    // Executar e analisar cada layer
    for (let layerIdx = 0; layerIdx <= 64; layerIdx++) {
        try {
            instance.exports.run_layer(layerIdx);
            
            // Ler parâmetros da layer
            const paramsBase = 1760576 + layerIdx * 112;
            const params = new Int32Array(memory.buffer, paramsBase, 28);
            
            const opType = params[0];
            const outPtr = params[4];
            const outH = params[26];
            const outW = params[27];
            const cout = params[8];
            
            const opNames = ['', 'CONV', 'DW', 'FC', 'ADD', 'MEAN', 'SOFTMAX'];
            
            // Calcular tamanho da saída
            let outputSize;
            if (opType === 6) {  // SOFTMAX
                outputSize = 1000;
            } else if (opType === 5) {  // MEAN
                outputSize = cout;
            } else if (opType === 3) {  // FC
                outputSize = cout;
            } else {  // CONV/DW/ADD
                outputSize = outH * outW * cout;
            }
            
            const sampleSize = Math.min(outputSize, 10000);
            const output = new Int8Array(memory.buffer, outPtr, sampleSize);
            
            const values = Array.from(output);
            const stats = {
                min: Math.min(...values),
                max: Math.max(...values),
                avg: values.reduce((a, b) => a + b, 0) / values.length,
                unique: new Set(values).size,
                sample: values.slice(0, 20)
            };
            
            const report = {
                layer: layerIdx,
                opType: opNames[opType] || `TYPE_${opType}`,
                shape: `${outH}x${outW}x${cout}`,
                stats: stats,
                healthy: stats.unique > 10  // Consideramos saudável se tiver >10 valores únicos
            };
            
            layerReport.push(report);
            
            // Print resumido
            const healthIcon = report.healthy ? '✅' : '⚠️';
            console.log(`${healthIcon} L${layerIdx.toString().padStart(2)} ${report.opType.padEnd(8)} ${report.shape.padEnd(15)} unique:${stats.unique.toString().padStart(4)} range:[${stats.min.toString().padStart(4)}, ${stats.max.toString().padStart(4)}]`);
            
            // Se detectar colapso, detalhar
            if (!report.healthy && layerIdx > 0) {
                console.log(`   ⚠️  POSSÍVEL PROBLEMA: Apenas ${stats.unique} valores únicos!`);
                console.log(`   Sample: [${stats.sample.join(', ')}]`);
            }
            
        } catch (e) {
            console.error(`\n❌ ERRO na Layer ${layerIdx}:`, e.message);
            break;
        }
    }
    
    console.log('\n═══════════════════════════════════════════════════════\n');
    
    // Análise final
    const unhealthyLayers = layerReport.filter(r => !r.healthy);
    
    console.log('📊 RESUMO:\n');
    console.log(`   Total de layers: ${layerReport.length}`);
    console.log(`   Layers saudáveis: ${layerReport.length - unhealthyLayers.length}`);
    console.log(`   Layers com problemas: ${unhealthyLayers.length}\n`);
    
    if (unhealthyLayers.length > 0) {
        console.log('⚠️  LAYERS COM POUCOS VALORES ÚNICOS:\n');
        unhealthyLayers.forEach(layer => {
            console.log(`   L${layer.layer} (${layer.opType}): ${layer.stats.unique} valores únicos`);
        });
        
        console.log('\n💡 DIAGNÓSTICO:\n');
        if (unhealthyLayers[0].layer === 0) {
            console.log('   O problema começa na LAYER 0!');
            console.log('   Possíveis causas:');
            console.log('   - Pesos ou bias incorretos');
            console.log('   - Multiplicadores de quantização errados');
            console.log('   - Problema na lógica de requantização\n');
        } else {
            console.log(`   O problema começa na LAYER ${unhealthyLayers[0].layer}`);
            console.log(`   Layers 0-${unhealthyLayers[0].layer - 1} parecem OK\n`);
        }
    } else {
        console.log('✅ TODAS as layers parecem estar funcionando!\n');
        console.log('   O problema deve estar na interpretação dos resultados.');
    }
    
    // Salvar relatório completo
    fs.writeFileSync('layer_trace.json', JSON.stringify({
        timestamp: new Date().toISOString(),
        layers: layerReport,
        summary: {
            total: layerReport.length,
            healthy: layerReport.length - unhealthyLayers.length,
            problematic: unhealthyLayers.length,
            firstProblem: unhealthyLayers[0]?.layer || null
        }
    }, null, 2));
    
    console.log('💾 Relatório completo salvo em: layer_trace.json\n');
    
})().catch(err => {
    console.error('❌ ERRO:', err.message);
    console.error(err.stack);
});