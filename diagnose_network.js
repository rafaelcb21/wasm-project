const fs = require('fs');

/**
 * Script de Diagnóstico da Rede MobileNetV2
 * Analisa cada camada para encontrar problemas
 * VERSÃO CORRIGIDA: sem conversão RGB565, sem stack overflow
 */

function analyzeInt8Array(arr, name) {
    const values = Array.from(arr);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    // Calcular média
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        sum += values[i];
    }
    const avg = sum / values.length;
    
    // Calcular desvio padrão
    let sumSquares = 0;
    for (let i = 0; i < values.length; i++) {
        const diff = values[i] - avg;
        sumSquares += diff * diff;
    }
    const std = Math.sqrt(sumSquares / values.length);
    
    // Contar saturação
    let saturatedMin = 0;
    let saturatedMax = 0;
    let zeros = 0;
    
    for (let i = 0; i < values.length; i++) {
        if (values[i] === -128) saturatedMin++;
        if (values[i] === 127) saturatedMax++;
        if (values[i] === 0) zeros++;
    }
    
    const saturatedTotal = saturatedMin + saturatedMax;
    const saturatedPercent = (saturatedTotal / values.length * 100).toFixed(2);
    const zerosPercent = (zeros / values.length * 100).toFixed(2);
    
    // Unique values (otimizado)
    const uniqueSet = new Set();
    for (let i = 0; i < values.length; i++) {
        uniqueSet.add(values[i]);
    }
    const uniqueValues = uniqueSet.size;
    
    return {
        name,
        count: values.length,
        min,
        max,
        avg: avg.toFixed(2),
        std: std.toFixed(2),
        saturatedMin,
        saturatedMax,
        saturatedTotal,
        saturatedPercent,
        zeros,
        zerosPercent,
        uniqueValues,
        distribution: getDistribution(values)
    };
}

function getDistribution(values) {
    const bins = {};
    for (let i = 0; i < values.length; i++) {
        const bin = Math.floor(values[i] / 10) * 10;
        bins[bin] = (bins[bin] || 0) + 1;
    }
    return bins;
}

function printAnalysis(analysis) {
    console.log(`\n📊 ${analysis.name}`);
    console.log('─'.repeat(70));
    console.log(`   Elementos: ${analysis.count}`);
    console.log(`   Range: [${analysis.min}, ${analysis.max}]`);
    console.log(`   Média: ${analysis.avg} | Desvio: ${analysis.std}`);
    console.log(`   Valores únicos: ${analysis.uniqueValues}/${analysis.count}`);
    console.log(`   Zeros: ${analysis.zeros} (${analysis.zerosPercent}%)`);
    console.log(`   Saturação: ${analysis.saturatedTotal} (${analysis.saturatedPercent}%)`);
    console.log(`      → -128: ${analysis.saturatedMin}`);
    console.log(`      → +127: ${analysis.saturatedMax}`);
    
    // Alertas
    if (parseFloat(analysis.saturatedPercent) > 10) {
        console.log(`   ⚠️  ALTA SATURAÇÃO! Mais de 10% dos valores estão saturados`);
    }
    if (parseFloat(analysis.zerosPercent) > 50) {
        console.log(`   ⚠️  MUITOS ZEROS! Mais de 50% dos valores são zero`);
    }
    if (analysis.uniqueValues < analysis.count * 0.1) {
        console.log(`   ⚠️  BAIXA DIVERSIDADE! Poucos valores únicos`);
    }
}

(async () => {
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║        DIAGNÓSTICO DETALHADO - MobileNetV2 WASM           ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');
    
    try {
        // ============================================================
        // 1. CARREGAR WASM
        // ============================================================
        console.log('📦 Carregando WASM...');
        const wasmBuffer = fs.readFileSync("main.wasm");
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const instance = wasmModule.instance;
        const memory = instance.exports.memory;
        
        console.log('✅ WASM carregado\n');
        
        // ============================================================
        // 2. CARREGAR IMAGEM (RGB888 direto, sem conversão!)
        // ============================================================
        console.log('📷 Carregando imagem RGB888...');
        const rawImage = fs.readFileSync("aviao_uint8.raw");
        
        const expectedSize = 224 * 224 * 3;
        console.log(`   Arquivo: aviao_uint8.raw`);
        console.log(`   Tamanho: ${rawImage.length} bytes`);
        console.log(`   Esperado: ${expectedSize} bytes (224×224×3)\n`);
        
        if (rawImage.length !== expectedSize) {
            throw new Error(`Tamanho incorreto! Esperado ${expectedSize}, encontrado ${rawImage.length}`);
        }
        
        // Analisar apenas uma amostra da imagem (primeiros 10000 pixels)
        console.log('Analisando amostra da imagem de entrada (primeiros 10000 pixels)...');
        const sampleSize = Math.min(10000, rawImage.length);
        const imgSample = new Int8Array(rawImage.buffer, 0, sampleSize);
        const imgAnalysis = analyzeInt8Array(imgSample, "IMAGEM DE ENTRADA (amostra)");
        printAnalysis(imgAnalysis);
        
        // Copiar imagem diretamente para memória WASM (RGB888)
        const inputPtr = 1767856;
        const memoryView = new Uint8Array(memory.buffer);
        memoryView.set(rawImage, inputPtr);
        
        console.log(`\n✅ Imagem RGB888 carregada diretamente no endereço 0x${inputPtr.toString(16)}\n`);
        
        // ============================================================
        // 3. EXECUTAR REDE
        // ============================================================
        console.log('⚙️  Executando inferência...\n');
        const startTime = Date.now();
        instance.exports.run_mobilenetv2();
        const endTime = Date.now();
        console.log(`✅ Inferência em ${endTime - startTime}ms\n`);
        
        // ============================================================
        // 4. ANALISAR SAÍDA FINAL
        // ============================================================
        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║                    ANÁLISE DA SAÍDA FINAL                  ║');
        console.log('╚════════════════════════════════════════════════════════════╝');
        
        const resultPtr = instance.exports.get_result_ptr();
        console.log(`   Endereço do resultado: 0x${resultPtr.toString(16)}\n`);
        
        const outputArray = new Int8Array(memory.buffer, resultPtr, 1000);
        
        const outputAnalysis = analyzeInt8Array(outputArray, "SAÍDA FINAL (1000 classes)");
        printAnalysis(outputAnalysis);
        
        // ============================================================
        // 5. ANÁLISE DETALHADA DA DISTRIBUIÇÃO
        // ============================================================
        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║                  DISTRIBUIÇÃO DE VALORES                   ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        const dist = outputAnalysis.distribution;
        const sortedBins = Object.keys(dist).map(Number).sort((a, b) => a - b);
        
        console.log('Bin    | Contagem | Percentual | Barra');
        console.log('─'.repeat(70));
        
        for (let bin of sortedBins) {
            const count = dist[bin];
            const percent = (count / 1000 * 100).toFixed(1);
            const barLength = Math.floor(count / 20);
            const bar = '█'.repeat(barLength);
            console.log(`${bin.toString().padStart(4)} | ${count.toString().padStart(8)} | ${percent.padStart(6)}% | ${bar}`);
        }
        
        // ============================================================
        // 6. ANÁLISE DOS TOP-20
        // ============================================================
        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║                    TOP-20 VALORES RAW                      ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        const indexed = [];
        for (let i = 0; i < outputArray.length; i++) {
            indexed.push({ val: outputArray[i], idx: i });
        }
        indexed.sort((a, b) => b.val - a.val);
        
        console.log('Rank | Index | Valor | Diferença do Top-1');
        console.log('─'.repeat(70));
        
        for (let i = 0; i < 20; i++) {
            const { val, idx } = indexed[i];
            const diff = val - indexed[0].val;
            console.log(`${(i+1).toString().padStart(4)} | ${idx.toString().padStart(5)} | ${val.toString().padStart(5)} | ${diff.toString().padStart(5)}`);
        }
        
        // Contar quantos têm valor máximo
        const maxValueCount = indexed.filter(x => x.val === indexed[0].val).length;
        console.log(`\n⚠️  ${maxValueCount} classes compartilham o valor máximo (${indexed[0].val})`);
        
        // ============================================================
        // 7. ANÁLISE DOS BOTTOM-20
        // ============================================================
        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║                   BOTTOM-20 VALORES RAW                    ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        console.log('Rank | Index | Valor');
        console.log('─'.repeat(70));
        
        for (let i = 980; i < 1000; i++) {
            const { val, idx } = indexed[i];
            console.log(`${(i+1).toString().padStart(4)} | ${idx.toString().padStart(5)} | ${val.toString().padStart(5)}`);
        }
        
        // ============================================================
        // 8. DIAGNÓSTICO FINAL
        // ============================================================
        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║                      DIAGNÓSTICO FINAL                     ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        const issues = [];
        
        if (parseFloat(outputAnalysis.saturatedPercent) > 10) {
            issues.push({
                severity: '🔴 CRÍTICO',
                issue: 'Alta saturação na saída',
                description: `${outputAnalysis.saturatedPercent}% dos valores estão saturados em +127 ou -128`,
                possibleCause: 'Problema na quantização da última camada (FC ou Softmax)',
                solution: 'Verificar parâmetros mul/shift da camada 66 (Softmax) e layer 65 (FC)'
            });
        }
        
        if (outputAnalysis.uniqueValues < 100) {
            issues.push({
                severity: '🔴 CRÍTICO',
                issue: 'Baixa diversidade de valores',
                description: `Apenas ${outputAnalysis.uniqueValues} valores únicos em 1000 classes`,
                possibleCause: 'Pesos ou bias da última camada podem estar errados',
                solution: 'Verificar arquivo de pesos da camada Fully Connected (layer 65)'
            });
        }
        
        if (maxValueCount > 100) {
            issues.push({
                severity: '🔴 CRÍTICO',
                issue: 'Múltiplas classes com valor máximo',
                description: `${maxValueCount} classes têm valor ${indexed[0].val}`,
                possibleCause: 'Overflow na última camada ou problema no Softmax',
                solution: 'Verificar implementação do Softmax e parâmetros de quantização'
            });
        }
        
        if (parseFloat(outputAnalysis.zerosPercent) > 50) {
            issues.push({
                severity: '🟡 ATENÇÃO',
                issue: 'Muitos zeros na saída',
                description: `${outputAnalysis.zerosPercent}% dos valores são zero`,
                possibleCause: 'Dead neurons ou problema na camada anterior',
                solution: 'Verificar layer 65 (FC) e layer 64 (Mean)'
            });
        }
        
        if (issues.length === 0) {
            console.log('✅ Nenhum problema crítico detectado na saída\n');
        } else {
            console.log(`🚨 ${issues.length} PROBLEMA(S) DETECTADO(S):\n`);
            
            issues.forEach((issue, i) => {
                console.log(`${issue.severity} Problema ${i+1}: ${issue.issue}`);
                console.log(`   Descrição: ${issue.description}`);
                console.log(`   Causa provável: ${issue.possibleCause}`);
                console.log(`   Solução: ${issue.solution}\n`);
            });
        }
        
        // ============================================================
        // 9. RECOMENDAÇÕES
        // ============================================================
        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║                       RECOMENDAÇÕES                        ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        console.log('1️⃣  VERIFICAR PARÂMETROS DA CAMADA SOFTMAX (Layer 66):');
        console.log('    • input_beta_mul (kh) - campo 9 da LayerParam');
        console.log('    • input_beta_shift (kw) - campo 10');
        console.log('    • diff_min (stride_h) - campo 11');
        console.log('    • integer_bits (stride_w) - campo 12');
        console.log('    • zX (pad_l) e zY (zy) - campos 17 e 25\n');
        
        console.log('2️⃣  VERIFICAR CAMADA FULLY CONNECTED (Layer 65):');
        console.log('    • Pesos (wptr) - devem ter 1280 × 1000 = 1,280,000 bytes');
        console.log('    • Bias (bias_ptr) - devem ter 1000 × 4 = 4,000 bytes');
        console.log('    • Multiplicadores (mul_ptr) - 1000 × 4 = 4,000 bytes Q31');
        console.log('    • Zero points (zx, zw, zy) - campos 23, 24, 25\n');
        
        console.log('3️⃣  TESTE COM IMAGENS SIMPLES:');
        console.log('    Execute: node generate_test_images.js');
        console.log('    Depois teste cada uma para ver se a rede responde\n');
        
        console.log('4️⃣  COMPARAR COM MODELO ORIGINAL:');
        console.log('    • Rodar a mesma imagem no TFLite/PyTorch');
        console.log('    • Comparar logits antes do Softmax');
        console.log('    • Verificar se a conversão int8 está correta\n');
        
        // ============================================================
        // 10. SALVAR DIAGNÓSTICO
        // ============================================================
        const diagnostic = {
            timestamp: new Date().toISOString(),
            executionTimeMs: endTime - startTime,
            inputAnalysis: imgAnalysis,
            outputAnalysis: outputAnalysis,
            maxValueCount: maxValueCount,
            top20: indexed.slice(0, 20).map((x, i) => ({
                rank: i + 1,
                classIndex: x.idx,
                rawValue: x.val
            })),
            bottom20: indexed.slice(980).map((x, i) => ({
                rank: 981 + i,
                classIndex: x.idx,
                rawValue: x.val
            })),
            issues: issues,
            distribution: dist
        };
        
        fs.writeFileSync('diagnostic_report.json', JSON.stringify(diagnostic, null, 2));
        
        // Salvar todos os valores em ordem (não ordenados)
        const allValuesText = [];
        for (let i = 0; i < outputArray.length; i++) {
            allValuesText.push(`${i}\t${outputArray[i]}`);
        }
        fs.writeFileSync('output_all_values.txt', allValuesText.join('\n'));
        
        // Salvar valores ordenados
        const sortedValuesText = indexed.map((x, i) => `${i+1}\t${x.idx}\t${x.val}`);
        fs.writeFileSync('output_sorted_values.txt', sortedValuesText.join('\n'));
        
        console.log('💾 Relatório salvo:');
        console.log('   ✅ diagnostic_report.json');
        console.log('   ✅ output_all_values.txt (valores por índice)');
        console.log('   ✅ output_sorted_values.txt (valores ordenados)\n');
        
        console.log('═'.repeat(70));
        console.log('DIAGNÓSTICO CONCLUÍDO');
        console.log('═'.repeat(70) + '\n');
        
    } catch (error) {
        console.error('\n❌ ERRO:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();