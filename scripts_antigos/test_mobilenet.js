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

function loadImageNetLabels(filename = 'imagenet_labels.txt') {
    try {
        const content = fs.readFileSync(filename, 'utf8');
        return content.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    } catch (e) {
        console.warn('⚠️  Arquivo de labels não encontrado');
        return Array.from({ length: 1000 }, (_, i) => `Classe_${i}`);
    }
}

// Função melhorada para calcular probabilidades aproximadas
function calculateProbabilities(outputArray) {
    // Converter int8 [-128, 127] para valores utilizáveis
    const values = Array.from(outputArray).map(v => v + 128); // Agora [0, 255]
    
    // Encontrar max para normalização tipo softmax
    const max = Math.max(...values);
    
    // Aplicar exp aproximado (para valores quantizados, usamos proporção direta)
    const expValues = values.map(v => {
        // Quanto mais próximo do max, maior o peso exponencial
        const diff = max - v;
        // Aproximação: exp(-diff/T) onde T é temperatura
        // Para int8, usamos peso linear decrescente
        return Math.max(0, 255 - diff * 2); // Penaliza diferença
    });
    
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    
    // Normalizar para probabilidades
    const probabilities = expValues.map(exp => exp / sumExp);
    
    return probabilities;
}

(async () => {
    console.log('🚀 MobileNetV2 - Classificação de Imagem\n');
    
    try {
        // 1. Carregar WASM
        const wasmBuffer = fs.readFileSync("main.wasm");
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const instance = wasmModule.instance;
        const memory = instance.exports.memory;
        const memoryView = new Uint8Array(memory.buffer);
        
        console.log('✅ Módulo WASM carregado');
        
        // 2. Carregar imagem
        console.log('📷 Carregando imagem...');
        const rawImage = fs.readFileSync("aviao_224x224x3.raw");
        
        if (rawImage.length !== 224 * 224 * 3) {
            throw new Error(`Tamanho incorreto!`);
        }
        
        const rgb565Image = rgb888ToRgb565(rawImage, 224, 224);
        const rgb565Bytes = new Uint8Array(rgb565Image.buffer);
        
        const inputPtr = 1767856;
        memoryView.set(rgb565Bytes, inputPtr);
        console.log('✅ Imagem carregada\n');
        
        // 3. Executar rede
        console.log('⚙️  Executando rede neural...');
        const startTime = Date.now();
        instance.exports.run_mobilenetv2();
        const endTime = Date.now();
        console.log(`✅ Inferência em ${endTime - startTime}ms\n`);
        
        // 4. Ler resultados
        const resultPtr = instance.exports.get_result_ptr();
        const outputArray = new Int8Array(memory.buffer, resultPtr, 1000);
        
        // 5. Calcular probabilidades
        console.log('📊 Calculando probabilidades...\n');
        const probabilities = calculateProbabilities(outputArray);
        
        // 6. Criar ranking
        const results = probabilities.map((prob, idx) => ({
            classIndex: idx,
            probability: prob,
            rawValue: outputArray[idx]
        })).sort((a, b) => b.probability - a.probability);
        
        // 7. Carregar labels
        const labels = loadImageNetLabels();
        
        // 8. Display Top-20
        console.log('═══════════════════════════════════════════════════════');
        console.log('                   🏆 RESULTADOS                        ');
        console.log('═══════════════════════════════════════════════════════\n');
        
        console.log('📋 TOP-20 PREDIÇÕES:\n');
        
        for (let i = 0; i < 20; i++) {
            const { classIndex, probability, rawValue } = results[i];
            const label = labels[classIndex] || `Classe_${classIndex}`;
            
            // Barra visual
            const barLength = Math.floor(probability * 50);
            const bar = '█'.repeat(barLength) + '░'.repeat(50 - barLength);
            
            const percentStr = (probability * 100).toFixed(2).padStart(6);
            
            console.log(`${(i + 1).toString().padStart(2)}. ${bar} ${percentStr}%`);
            console.log(`    [${classIndex.toString().padStart(4)}] ${label}`);
            console.log(`    Raw: ${rawValue.toString().padStart(4)}\n`);
        }
        
        console.log('═══════════════════════════════════════════════════════\n');
        
        // 9. Estatísticas detalhadas
        const rawValues = Array.from(outputArray);
        const stats = {
            raw: {
                min: Math.min(...rawValues),
                max: Math.max(...rawValues),
                avg: rawValues.reduce((a, b) => a + b, 0) / rawValues.length,
                uniqueValues: new Set(rawValues).size
            },
            probabilities: {
                top1: results[0].probability,
                top5_sum: results.slice(0, 5).reduce((sum, r) => sum + r.probability, 0),
                entropy: -results.reduce((sum, r) => {
                    return sum + (r.probability > 0 ? r.probability * Math.log2(r.probability) : 0);
                }, 0)
            }
        };
        
        console.log('📊 ESTATÍSTICAS DETALHADAS:\n');
        console.log('   Valores Raw (int8):');
        console.log(`      Range: [${stats.raw.min}, ${stats.raw.max}]`);
        console.log(`      Média: ${stats.raw.avg.toFixed(2)}`);
        console.log(`      Valores únicos: ${stats.raw.uniqueValues}/1000`);
        console.log('\n   Probabilidades:');
        console.log(`      Confiança Top-1: ${(stats.probabilities.top1 * 100).toFixed(2)}%`);
        console.log(`      Soma Top-5: ${(stats.probabilities.top5_sum * 100).toFixed(2)}%`);
        console.log(`      Entropia: ${stats.probabilities.entropy.toFixed(2)} bits`);
        console.log(`      (Menor entropia = maior certeza)`);
        console.log(`\n   Performance:`);
        console.log(`      Tempo de inferência: ${endTime - startTime}ms\n`);
        
        // 10. Análise de certeza
        console.log('🎯 ANÁLISE DE CERTEZA:\n');
        if (stats.probabilities.top1 > 0.5) {
            console.log('   ✅ Alta confiança na predição');
        } else if (stats.probabilities.top1 > 0.2) {
            console.log('   ⚠️  Confiança moderada');
        } else {
            console.log('   ⚠️  Baixa confiança - múltiplas classes possíveis');
        }
        
        if (stats.probabilities.entropy < 3) {
            console.log('   ✅ Baixa entropia - classificação clara');
        } else if (stats.probabilities.entropy < 6) {
            console.log('   ⚠️  Entropia média');
        } else {
            console.log('   ⚠️  Alta entropia - classificação incerta');
        }
        
        console.log('\n═══════════════════════════════════════════════════════\n');
        
        // 11. Salvar resultados
        const outputData = {
            timestamp: new Date().toISOString(),
            executionTimeMs: endTime - startTime,
            image: {
                filename: "aviao_224x224x3.raw",
                dimensions: { width: 224, height: 224, channels: 3 }
            },
            top20: results.slice(0, 20).map((r, i) => ({
                rank: i + 1,
                classIndex: r.classIndex,
                className: labels[r.classIndex] || `Classe_${r.classIndex}`,
                probability: r.probability,
                probabilityPercent: (r.probability * 100).toFixed(2) + '%',
                rawValue: r.rawValue
            })),
            statistics: stats
        };
        
        fs.writeFileSync('resultado_final.json', JSON.stringify(outputData, null, 2));
        fs.writeFileSync('output_raw.bin', Buffer.from(outputArray));
        
        console.log('💾 Resultados salvos:');
        console.log('   - resultado_final.json');
        console.log('   - output_raw.bin\n');
        
        console.log('✨ Classificação concluída!\n');
        
    } catch (error) {
        console.error('\n❌ ERRO:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();