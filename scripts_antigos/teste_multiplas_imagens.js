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

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sumExps);
}

function loadImageNetLabels(filename = 'imagenet_labels.txt') {
    try {
        const content = fs.readFileSync(filename, 'utf8');
        return content.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    } catch (e) {
        return Array.from({ length: 1000 }, (_, i) => `Classe_${i}`);
    }
}

(async () => {
    console.log('🧪 TESTE COM DIFERENTES IMAGENS\n');
    
    const wasmBuffer = fs.readFileSync("main.wasm");
    const wasmModule = await WebAssembly.instantiate(wasmBuffer);
    const instance = wasmModule.instance;
    const memory = instance.exports.memory;
    const labels = loadImageNetLabels();
    
    // Testar 4 cenários diferentes
    const tests = [
        { name: 'Imagem Original (aviao.raw)', file: 'aviao_224x224x3.raw' },
        { name: 'Imagem Preta', color: [0, 0, 0] },
        { name: 'Imagem Branca', color: [255, 255, 255] },
        { name: 'Imagem Cinza (128)', color: [128, 128, 128] }
    ];
    
    const results = [];
    
    for (const test of tests) {
        console.log(`═══════════════════════════════════════════════════════`);
        console.log(`🖼️  Testando: ${test.name}\n`);
        
        let rgb888Buffer;
        
        if (test.file) {
            // Carregar arquivo
            rgb888Buffer = fs.readFileSync(test.file);
            console.log(`   Carregado: ${rgb888Buffer.length} bytes`);
        } else {
            // Criar imagem sólida
            rgb888Buffer = Buffer.alloc(224 * 224 * 3);
            for (let i = 0; i < 224 * 224; i++) {
                rgb888Buffer[i * 3 + 0] = test.color[0];
                rgb888Buffer[i * 3 + 1] = test.color[1];
                rgb888Buffer[i * 3 + 2] = test.color[2];
            }
            console.log(`   Criada: cor RGB(${test.color.join(', ')})`);
        }
        
        // Converter para RGB565
        const rgb565Image = rgb888ToRgb565(rgb888Buffer, 224, 224);
        const rgb565Bytes = new Uint8Array(rgb565Image.buffer);
        
        // Carregar na memória
        const inputPtr = 1767856;
        new Uint8Array(memory.buffer).set(rgb565Bytes, inputPtr);
        
        console.log(`   Executando rede...`);
        const startTime = Date.now();
        
        // Executar até Layer 63
        for (let i = 0; i <= 63; i++) {
            instance.exports.run_layer(i);
        }
        
        const execTime = Date.now() - startTime;
        
        // Ler logits
        const layer64ParamsBase = 1760576 + 64 * 112;
        const layer64Params = new Int32Array(memory.buffer, layer64ParamsBase, 28);
        const fcOutputPtr = layer64Params[15];
        const logitsArray = new Int8Array(memory.buffer, fcOutputPtr, 1000);
        
        // Processar
        const logitsFloat = Array.from(logitsArray).map(x => x / 10.0);
        const probabilities = softmax(logitsFloat);
        
        const ranked = probabilities.map((prob, idx) => ({
            classIndex: idx,
            probability: prob,
            logit: logitsArray[idx]
        })).sort((a, b) => b.probability - a.probability);
        
        // Mostrar Top-5
        console.log(`   ✅ Concluído em ${execTime}ms\n`);
        console.log(`   🏆 TOP-5:\n`);
        
        for (let i = 0; i < 5; i++) {
            const { classIndex, probability, logit } = ranked[i];
            const label = labels[classIndex] || `Classe_${classIndex}`;
            console.log(`      ${i + 1}. [${classIndex}] ${label}`);
            console.log(`         ${(probability * 100).toFixed(2)}% (logit: ${logit})\n`);
        }
        
        // Estatísticas
        const logitValues = Array.from(logitsArray);
        const stats = {
            logitRange: [Math.min(...logitValues), Math.max(...logitValues)],
            logitAvg: logitValues.reduce((a, b) => a + b, 0) / 1000,
            top1Prob: ranked[0].probability,
            entropy: -ranked.reduce((sum, r) => 
                sum + (r.probability > 1e-10 ? r.probability * Math.log2(r.probability) : 0), 0)
        };
        
        console.log(`   📊 Estatísticas:`);
        console.log(`      Logit range: [${stats.logitRange[0]}, ${stats.logitRange[1]}]`);
        console.log(`      Logit avg: ${stats.logitAvg.toFixed(2)}`);
        console.log(`      Top-1 conf: ${(stats.top1Prob * 100).toFixed(2)}%`);
        console.log(`      Entropia: ${stats.entropy.toFixed(2)} bits\n`);
        
        results.push({
            test: test.name,
            executionTimeMs: execTime,
            top5: ranked.slice(0, 5).map(r => ({
                classIndex: r.classIndex,
                className: labels[r.classIndex],
                probability: r.probability,
                logit: r.logit
            })),
            statistics: stats
        });
    }
    
    console.log(`═══════════════════════════════════════════════════════\n`);
    
    // Salvar comparação
    fs.writeFileSync('teste_comparacao.json', JSON.stringify({
        timestamp: new Date().toISOString(),
        note: "Comparação entre diferentes imagens de entrada",
        tests: results
    }, null, 2));
    
    console.log('💾 Resultados salvos em: teste_comparacao.json\n');
    
    // Análise comparativa
    console.log('🔍 ANÁLISE COMPARATIVA:\n');
    
    const originalTest = results[0];
    const blackTest = results.find(r => r.test.includes('Preta'));
    const whiteTest = results.find(r => r.test.includes('Branca'));
    
    console.log(`   Imagem Original:`);
    console.log(`      Top-1: ${originalTest.top5[0].className}`);
    console.log(`      Confiança: ${(originalTest.top5[0].probability * 100).toFixed(2)}%`);
    console.log(`      Entropia: ${originalTest.statistics.entropy.toFixed(2)}\n`);
    
    if (originalTest.statistics.entropy > 9) {
        console.log('   ⚠️  PROBLEMA IDENTIFICADO:');
        console.log('      A entropia muito alta indica que a rede está');
        console.log('      completamente incerta sobre a classificação.\n');
        console.log('   💡 POSSÍVEIS CAUSAS:');
        console.log('      1. Arquivo aviao.raw não contém uma imagem válida');
        console.log('      2. Formato do arquivo está incorreto');
        console.log('      3. Canais RGB estão em ordem errada (BGR?)');
        console.log('      4. Pesos da rede não correspondem ao formato esperado\n');
    }
    
    console.log('═══════════════════════════════════════════════════════\n');
    
})().catch(err => {
    console.error('❌ ERRO:', err.message);
    console.error(err.stack);
});