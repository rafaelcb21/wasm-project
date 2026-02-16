const fs = require('fs');

/**
 * Analisador de Multiplicadores Q31 - Layer 65 (Fully Connected)
 * Extrai e valida os multiplicadores da camada final
 */

console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║      ANALISADOR DE MULTIPLICADORES Q31 - LAYER 65         ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

(async () => {
    try {
        // Carregar WASM
        console.log('📦 Carregando WASM...');
        const wasmBuffer = fs.readFileSync("main.wasm");
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const instance = wasmModule.instance;
        const memory = instance.exports.memory;
        const memoryView = new DataView(memory.buffer);
        
        console.log('✅ WASM carregado\n');
        
        // Parâmetros da Layer 65
        const PARAMS_BASE = 1760576;
        const LP_SIZE = 112;
        const layer65_idx = 65;
        
        // Calcular endereço base da Layer 65
        const layer65_base = PARAMS_BASE + (layer65_idx * LP_SIZE);
        
        console.log('📊 Layer 65 (Fully Connected):');
        console.log(`   Base address: ${layer65_base} (0x${layer65_base.toString(16)})\n`);
        
        // Ler campos importantes (cada campo é i32, 4 bytes)
        const cin = memoryView.getInt32(layer65_base + (7 * 4), true);  // campo 7
        const cout = memoryView.getInt32(layer65_base + (8 * 4), true); // campo 8
        const wptr = memoryView.getInt32(layer65_base + (19 * 4), true); // campo 19
        const bias_ptr = memoryView.getInt32(layer65_base + (20 * 4), true); // campo 20
        const mul_ptr = memoryView.getInt32(layer65_base + (21 * 4), true); // campo 21
        const zx = memoryView.getInt32(layer65_base + (23 * 4), true); // campo 23
        const zw = memoryView.getInt32(layer65_base + (24 * 4), true); // campo 24
        const zy = memoryView.getInt32(layer65_base + (25 * 4), true); // campo 25
        
        console.log('🔍 Parâmetros da Layer:');
        console.log(`   cin (entrada):    ${cin}`);
        console.log(`   cout (saída):     ${cout}`);
        console.log(`   wptr (pesos):     ${wptr} (0x${wptr.toString(16)})`);
        console.log(`   bias_ptr:         ${bias_ptr} (0x${bias_ptr.toString(16)})`);
        console.log(`   mul_ptr:          ${mul_ptr} (0x${mul_ptr.toString(16)})`);
        console.log(`   zx (zp entrada):  ${zx}`);
        console.log(`   zw (zp pesos):    ${zw}`);
        console.log(`   zy (zp saída):    ${zy}\n`);
        
        // Ler multiplicadores Q31
        console.log('📈 Analisando multiplicadores Q31...\n');
        
        const multipliers = [];
        for (let i = 0; i < cout; i++) {
            const mul = memoryView.getInt32(mul_ptr + (i * 4), true);
            multipliers.push(mul);
        }
        
        // Estatísticas dos multiplicadores
        const mulStats = {
            min: Math.min(...multipliers),
            max: Math.max(...multipliers),
            avg: multipliers.reduce((a, b) => a + b, 0) / multipliers.length,
            median: multipliers.slice().sort((a, b) => a - b)[Math.floor(multipliers.length / 2)],
            zeros: multipliers.filter(m => m === 0).length,
            negative: multipliers.filter(m => m < 0).length,
            positive: multipliers.filter(m => m > 0).length,
            veryLarge: multipliers.filter(m => Math.abs(m) > 2147483647 * 0.9).length,
            unique: new Set(multipliers).size
        };
        
        console.log('📊 ESTATÍSTICAS DOS MULTIPLICADORES:\n');
        console.log(`   Total:           ${cout}`);
        console.log(`   Valores únicos:  ${mulStats.unique}`);
        console.log(`   Range:           [${mulStats.min.toLocaleString()}, ${mulStats.max.toLocaleString()}]`);
        console.log(`   Média:           ${mulStats.avg.toLocaleString('en', {maximumFractionDigits: 0})}`);
        console.log(`   Mediana:         ${mulStats.median.toLocaleString()}`);
        console.log(`   Zeros:           ${mulStats.zeros}`);
        console.log(`   Negativos:       ${mulStats.negative}`);
        console.log(`   Positivos:       ${mulStats.positive}`);
        console.log(`   Muito grandes:   ${mulStats.veryLarge} (> 90% de INT32_MAX)\n`);
        
        // Análise dos multiplicadores
        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║                      ANÁLISE CRÍTICA                       ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        const issues = [];
        
        // Verificar se multiplicadores são muito grandes
        const Q31_MAX = 2147483647; // 2^31 - 1
        const tooLarge = multipliers.filter(m => Math.abs(m) > Q31_MAX * 0.5);
        
        if (tooLarge.length > cout * 0.1) {
            issues.push({
                severity: '🔴 CRÍTICO',
                problem: 'Multiplicadores muito grandes',
                description: `${tooLarge.length} multiplicadores (${(tooLarge.length/cout*100).toFixed(1)}%) são > 50% de INT32_MAX`,
                impact: 'Causará overflow durante multiply_by_quantized_multiplier',
                solution: 'Multiplicadores Q31 devem estar na faixa de 10^8 a 10^9 para MobileNetV2'
            });
        }
        
        // Verificar distribuição
        if (mulStats.unique < cout * 0.1) {
            issues.push({
                severity: '🟡 ATENÇÃO',
                problem: 'Baixa diversidade de multiplicadores',
                description: `Apenas ${mulStats.unique} valores únicos em ${cout} classes`,
                impact: 'Pode indicar arquivo de pesos corrompido ou mal configurado',
                solution: 'Verificar processo de quantização do modelo'
            });
        }
        
        // Verificar zeros
        if (mulStats.zeros > 0) {
            issues.push({
                severity: '🔴 CRÍTICO',
                problem: 'Multiplicadores com valor zero',
                description: `${mulStats.zeros} multiplicadores são exatamente 0`,
                impact: 'Essas classes sempre terão saída = zy (zero point)',
                solution: 'Nenhum multiplicador deveria ser zero'
            });
        }
        
        // Mostrar issues
        if (issues.length > 0) {
            console.log(`🚨 ${issues.length} PROBLEMA(S) DETECTADO(S):\n`);
            issues.forEach((issue, i) => {
                console.log(`${issue.severity} Problema ${i+1}: ${issue.problem}`);
                console.log(`   Descrição: ${issue.description}`);
                console.log(`   Impacto: ${issue.impact}`);
                console.log(`   Solução: ${issue.solution}\n`);
            });
        } else {
            console.log('✅ Multiplicadores dentro dos parâmetros normais\n');
        }
        
        // Mostrar Top-10 maiores e menores
        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║               TOP-10 MULTIPLICADORES (MAIORES)             ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        const indexed = multipliers.map((m, i) => ({ mul: m, idx: i }));
        indexed.sort((a, b) => b.mul - a.mul);
        
        console.log('Rank | Classe | Multiplicador      | % de INT32_MAX');
        console.log('─'.repeat(70));
        for (let i = 0; i < Math.min(10, indexed.length); i++) {
            const { mul, idx } = indexed[i];
            const percent = (Math.abs(mul) / Q31_MAX * 100).toFixed(2);
            console.log(`${(i+1).toString().padStart(4)} | ${idx.toString().padStart(6)} | ${mul.toString().padStart(18)} | ${percent.padStart(6)}%`);
        }
        
        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║               TOP-10 MULTIPLICADORES (MENORES)             ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        indexed.sort((a, b) => a.mul - b.mul);
        
        console.log('Rank | Classe | Multiplicador      | % de INT32_MAX');
        console.log('─'.repeat(70));
        for (let i = 0; i < Math.min(10, indexed.length); i++) {
            const { mul, idx } = indexed[i];
            const percent = (Math.abs(mul) / Q31_MAX * 100).toFixed(2);
            console.log(`${(i+1).toString().padStart(4)} | ${idx.toString().padStart(6)} | ${mul.toString().padStart(18)} | ${percent.padStart(6)}%`);
        }
        
        // Análise de bias
        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║                     ANÁLISE DOS BIAS                       ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        const biases = [];
        for (let i = 0; i < cout; i++) {
            const bias = memoryView.getInt32(bias_ptr + (i * 4), true);
            biases.push(bias);
        }
        
        const biasStats = {
            min: Math.min(...biases),
            max: Math.max(...biases),
            avg: biases.reduce((a, b) => a + b, 0) / biases.length,
            median: biases.slice().sort((a, b) => a - b)[Math.floor(biases.length / 2)],
            zeros: biases.filter(b => b === 0).length,
            unique: new Set(biases).size
        };
        
        console.log(`   Total:           ${cout}`);
        console.log(`   Valores únicos:  ${biasStats.unique}`);
        console.log(`   Range:           [${biasStats.min.toLocaleString()}, ${biasStats.max.toLocaleString()}]`);
        console.log(`   Média:           ${biasStats.avg.toLocaleString('en', {maximumFractionDigits: 0})}`);
        console.log(`   Mediana:         ${biasStats.median.toLocaleString()}`);
        console.log(`   Zeros:           ${biasStats.zeros}\n`);
        
        // Verificar bias extremos
        const extremeBias = biases.filter(b => Math.abs(b) > 1000000);
        if (extremeBias.length > 0) {
            console.log(`⚠️  ${extremeBias.length} bias com valores extremos (> ±1,000,000)`);
            console.log(`   Isso pode contribuir para saturação!\n`);
        }
        
        // Salvar dados
        const analysis = {
            timestamp: new Date().toISOString(),
            layer: 65,
            parameters: {
                cin, cout, wptr, bias_ptr, mul_ptr, zx, zw, zy
            },
            multipliers: {
                statistics: mulStats,
                top10_largest: indexed.slice(-10).reverse().map(x => ({ class: x.idx, value: x.mul })),
                top10_smallest: indexed.slice(0, 10).map(x => ({ class: x.idx, value: x.mul })),
                all_values: multipliers
            },
            biases: {
                statistics: biasStats,
                all_values: biases
            },
            issues: issues
        };
        
        fs.writeFileSync('layer65_analysis.json', JSON.stringify(analysis, null, 2));
        
        console.log('💾 Análise salva: layer65_analysis.json\n');
        
        // Recomendações finais
        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║                     RECOMENDAÇÕES FINAIS                   ║');
        console.log('╚════════════════════════════════════════════════════════════╝\n');
        
        if (tooLarge.length > 0) {
            console.log('🔧 AÇÃO NECESSÁRIA:');
            console.log('   Os multiplicadores Q31 estão causando overflow!\n');
            console.log('   Fórmula correta para Q31:');
            console.log('   M = (scale_input × scale_weights) / scale_output × 2^31\n');
            console.log('   Para MobileNetV2 típico:');
            console.log('   - scale_input: 0.003 - 0.01');
            console.log('   - scale_weights: 0.001 - 0.01');
            console.log('   - scale_output: 0.01 - 0.1');
            console.log('   - Resultado: M ≈ 10^8 - 10^9\n');
            console.log('   Seus multiplicadores estão muito maiores que isso!');
            console.log('   Solução: Re-quantizar o modelo com scales corretos.\n');
        } else {
            console.log('✅ Multiplicadores parecem estar na faixa correta.\n');
            console.log('   Se ainda há saturação, o problema pode estar em:');
            console.log('   1. Valores de entrada (Mean layer)');
            console.log('   2. Zero points incorretos');
            console.log('   3. Softmax mal configurado\n');
        }
        
        console.log('═'.repeat(70));
        console.log('ANÁLISE CONCLUÍDA');
        console.log('═'.repeat(70) + '\n');
        
    } catch (error) {
        console.error('\n❌ ERRO:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();
