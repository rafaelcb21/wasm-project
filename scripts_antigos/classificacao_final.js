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

// Softmax real para converter logits em probabilidades
function softmax(logits) {
    // Encontrar max para estabilidade numérica
    const maxLogit = Math.max(...logits);
    
    // Calcular exp de cada valor (subtraindo max)
    const exps = logits.map(x => Math.exp(x - maxLogit));
    
    // Somar todos os exp
    const sumExps = exps.reduce((a, b) => a + b, 0);
    
    // Normalizar
    return exps.map(exp => exp / sumExps);
}

(async () => {
    console.log('🚀 MobileNetV2 - Classificação de Imagem\n');
    console.log('   🔧 Usando saída da FC (ignorando Softmax quantizado)\n');
    
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
        
        // 3. Executar rede ATÉ A LAYER 63 (FC, antes do Softmax)
        console.log('⚙️  Executando rede neural (layers 0-63)...');
        const startTime = Date.now();
        
        for (let i = 0; i <= 63; i++) {
            instance.exports.run_layer(i);
        }
        
        const endTime = Date.now();
        console.log(`✅ Inferência em ${endTime - startTime}ms\n`);
        
        // 4. Ler saída da FC (Layer 63) - IGNORAR Layer 64 (Softmax ruim)
        // Obter in_ptr da Layer 64, que é a saída da Layer 63
        const layer64ParamsBase = 1760576 + 64 * 112;
        const layer64Params = new Int32Array(memory.buffer, layer64ParamsBase, 28);
        const fcOutputPtr = layer64Params[15]; // pad_t = input_ptr do softmax
        
        const logitsArray = new Int8Array(memory.buffer, fcOutputPtr, 1000);
        
        console.log('📊 Processando logits da FC...\n');
        
        // 5. Converter int8 logits para float e aplicar softmax
        const logitsFloat = Array.from(logitsArray).map(x => x / 10.0); // Escala aproximada
        const probabilities = softmax(logitsFloat);
        
        // 6. Criar ranking
        const results = probabilities.map((prob, idx) => ({
            classIndex: idx,
            probability: prob,
            logit: logitsArray[idx]
        })).sort((a, b) => b.probability - a.probability);
        
        // 7. Carregar labels
        const labels = loadImageNetLabels();
        
        // 8. Display Top-20
        console.log('═══════════════════════════════════════════════════════');
        console.log('                   🏆 RESULTADOS                        ');
        console.log('═══════════════════════════════════════════════════════\n');
        
        console.log('📋 TOP-20 PREDIÇÕES:\n');
        
        for (let i = 0; i < 20; i++) {
            const { classIndex, probability, logit } = results[i];
            const label = labels[classIndex] || `Classe_${classIndex}`;
            
            // Barra visual
            const barLength = Math.floor(probability * 50);
            const bar = '█'.repeat(barLength) + '░'.repeat(50 - barLength);
            
            const percentStr = (probability * 100).toFixed(2).padStart(6);
            
            console.log(`${(i + 1).toString().padStart(2)}. ${bar} ${percentStr}%`);
            console.log(`    [${classIndex.toString().padStart(4)}] ${label}`);
            console.log(`    Logit: ${logit}\n`);
        }
        
        console.log('═══════════════════════════════════════════════════════\n');
        
        // 9. Estatísticas
        const logitValues = Array.from(logitsArray);
        const stats = {
            logits: {
                min: Math.min(...logitValues),
                max: Math.max(...logitValues),
                avg: logitValues.reduce((a, b) => a + b, 0) / logitValues.length,
                uniqueValues: new Set(logitValues).size
            },
            probabilities: {
                top1: results[0].probability,
                top5_sum: results.slice(0, 5).reduce((sum, r) => sum + r.probability, 0),
                entropy: -results.reduce((sum, r) => {
                    return sum + (r.probability > 1e-10 ? r.probability * Math.log2(r.probability) : 0);
                }, 0)
            }
        };
        
        console.log('📊 ESTATÍSTICAS:\n');
        console.log('   Logits (saída FC):');
        console.log(`      Range: [${stats.logits.min}, ${stats.logits.max}]`);
        console.log(`      Média: ${stats.logits.avg.toFixed(2)}`);
        console.log(`      Valores únicos: ${stats.logits.uniqueValues}/1000`);
        console.log('\n   Probabilidades (após softmax):');
        console.log(`      Confiança Top-1: ${(stats.probabilities.top1 * 100).toFixed(2)}%`);
        console.log(`      Soma Top-5: ${(stats.probabilities.top5_sum * 100).toFixed(2)}%`);
        console.log(`      Entropia: ${stats.probabilities.entropy.toFixed(2)} bits`);
        console.log(`\n   Performance:`);
        console.log(`      Tempo: ${endTime - startTime}ms (sem softmax quantizado)\n`);
        
        // 10. Análise de certeza
        console.log('🎯 ANÁLISE:\n');
        if (stats.probabilities.top1 > 0.5) {
            console.log('   ✅ ALTA confiança na predição');
        } else if (stats.probabilities.top1 > 0.2) {
            console.log('   ⚠️  Confiança MODERADA');
        } else {
            console.log('   ⚠️  BAIXA confiança');
        }
        
        if (stats.probabilities.entropy < 3) {
            console.log('   ✅ Baixa entropia - classificação CLARA');
        } else if (stats.probabilities.entropy < 6) {
            console.log('   ⚠️  Entropia média');
        } else {
            console.log('   ⚠️  Alta entropia - INCERTA');
        }
        
        console.log('\n═══════════════════════════════════════════════════════\n');
        
        // 11. Salvar resultados
        const outputData = {
            timestamp: new Date().toISOString(),
            executionTimeMs: endTime - startTime,
            note: "Usando saída da FC (Layer 63), ignorando Softmax quantizado mal parametrizado",
            image: {
                filename: "aviao_224x224x3.raw",
                dimensions: { width: 224, height: 224, channels: 3 }
            },
            top20: results.slice(0, 20).map((r, i) => ({
                rank: i + 1,
                classIndex: r.classIndex,
                className: labels[r.classIndex] || `Classe_${r.classIndex}`,
                probability: r.probability,
                probabilityPercent: (r.probability * 100).toFixed(4) + '%',
                logit: r.logit
            })),
            statistics: stats
        };
        
        fs.writeFileSync('resultado_correto.json', JSON.stringify(outputData, null, 2));
        fs.writeFileSync('logits_fc.bin', Buffer.from(logitsArray));
        
        console.log('💾 Resultados salvos:');
        console.log('   - resultado_correto.json');
        console.log('   - logits_fc.bin\n');
        
        console.log('✨ Classificação concluída!\n');
        
    } catch (error) {
        console.error('\n❌ ERRO:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();