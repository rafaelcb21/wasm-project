const fs = require('fs');

(async () => {
    console.log('🔍 VERIFICAÇÃO DOS PESOS DA REDE\n');
    
    const diagnosticResults = {
        timestamp: new Date().toISOString(),
        layer0: {},
        weights: {},
        bias: {},
        multipliers: {},
        syntheticTest: {},
        diagnosis: []
    };
    
    try {
        const wasmBuffer = fs.readFileSync("main.wasm");
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const instance = wasmModule.instance;
        const memory = instance.exports.memory;
        
        // Verificar Layer 0 (primeira conv)
        const layer0ParamsBase = 1760576;
        const layer0Params = new Int32Array(memory.buffer, layer0ParamsBase, 28);
        
        diagnosticResults.layer0 = {
            wptr: layer0Params[19],
            bias_ptr: layer0Params[20],
            mul_ptr: layer0Params[21],
            cin: layer0Params[7],
            cout: layer0Params[8],
            kh: layer0Params[9],
            kw: layer0Params[10],
            in_ptr: layer0Params[3],
            out_ptr: layer0Params[4]
        };
        
        console.log('📋 LAYER 0 (Primeira CONV) - Parâmetros:\n');
        console.log(`   wptr (pesos): ${layer0Params[19]}`);
        console.log(`   bias_ptr: ${layer0Params[20]}`);
        console.log(`   cin: ${layer0Params[7]}, cout: ${layer0Params[8]}`);
        console.log(`   kh: ${layer0Params[9]}, kw: ${layer0Params[10]}`);
        
        // Ler alguns pesos
        const wptr = layer0Params[19];
        const weights = new Int8Array(memory.buffer, wptr, 100);
        
        console.log('\n📊 Primeiros 100 pesos:\n');
        console.log(`   Sample: [${Array.from(weights).slice(0, 20).join(', ')}]`);
        
        const weightStats = {
            min: Math.min(...Array.from(weights)),
            max: Math.max(...Array.from(weights)),
            avg: Array.from(weights).reduce((a, b) => a + b, 0) / weights.length,
            zeros: Array.from(weights).filter(w => w === 0).length,
            unique: new Set(Array.from(weights)).size,
            sample: Array.from(weights).slice(0, 20)
        };
        
        diagnosticResults.weights = weightStats;
        
        console.log('\n   Estatísticas:');
        console.log(`      Min: ${weightStats.min}`);
        console.log(`      Max: ${weightStats.max}`);
        console.log(`      Avg: ${weightStats.avg.toFixed(2)}`);
        console.log(`      Zeros: ${weightStats.zeros}/100`);
        console.log(`      Valores únicos: ${weightStats.unique}`);
        
        // Verificar bias
        const biasPtr = layer0Params[20];
        const bias = new Int32Array(memory.buffer, biasPtr, 16); // 16 canais de saída
        
        diagnosticResults.bias = {
            values: Array.from(bias),
            nonZero: Array.from(bias).filter(b => b !== 0).length
        };
        
        console.log('\n📊 Bias (primeiros 16):\n');
        console.log(`   [${Array.from(bias).join(', ')}]`);
        
        // Verificar multiplicadores
        const mulPtr = layer0Params[21];
        const multipliers = new Int32Array(memory.buffer, mulPtr, 16);
        
        diagnosticResults.multipliers = {
            values: Array.from(multipliers),
            nonZero: Array.from(multipliers).filter(m => m !== 0).length
        };
        
        console.log('\n📊 Multiplicadores Q31 (primeiros 16):\n');
        console.log(`   [${Array.from(multipliers).join(', ')}]`);
        
        // DIAGNÓSTICO
        console.log('\n═══════════════════════════════════════════════════════');
        console.log('🔍 DIAGNÓSTICO:\n');
        
        if (weightStats.zeros > 50) {
            const msg = '❌ PROBLEMA: Muitos zeros nos pesos! Os pesos não foram carregados corretamente.';
            console.log(msg);
            diagnosticResults.diagnosis.push({ severity: 'error', category: 'weights', message: msg });
        } else if (weightStats.unique < 10) {
            const msg = '❌ PROBLEMA: Poucos valores únicos! Os pesos parecem não ter variação suficiente.';
            console.log(msg);
            diagnosticResults.diagnosis.push({ severity: 'error', category: 'weights', message: msg });
        } else {
            const msg = '✅ Pesos parecem OK (variedade suficiente)';
            console.log(msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'ok', category: 'weights', message: msg });
        }
        
        const biasNonZero = Array.from(bias).filter(b => b !== 0).length;
        if (biasNonZero === 0) {
            const msg = '❌ PROBLEMA: Todos os bias são zero!';
            console.log(msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'error', category: 'bias', message: msg });
        } else {
            const msg = `✅ Bias OK (${biasNonZero}/16 não-zeros)`;
            console.log(msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'ok', category: 'bias', message: msg });
        }
        
        const mulNonZero = Array.from(multipliers).filter(m => m !== 0).length;
        if (mulNonZero === 0) {
            const msg = '❌ PROBLEMA: Todos os multiplicadores são zero!';
            console.log(msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'error', category: 'multipliers', message: msg });
        } else {
            const msg = `✅ Multiplicadores OK (${mulNonZero}/16 não-zeros)`;
            console.log(msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'ok', category: 'multipliers', message: msg });
        }
        
        console.log('═══════════════════════════════════════════════════════\n');
        
        // Testar com entrada sintética
        console.log('🧪 TESTE COM ENTRADA SINTÉTICA:\n');
        console.log('   Criando imagem de teste (gradiente)...');
        
        const testInput = new Uint16Array(224 * 224);
        for (let i = 0; i < 224 * 224; i++) {
            // Criar gradiente RGB565
            const val = (i % 224) / 224;
            const r5 = Math.floor(val * 31);
            const g6 = Math.floor(val * 63);
            const b5 = Math.floor(val * 31);
            testInput[i] = (b5 << 11) | (g6 << 5) | r5;
        }
        
        const inputPtr = 1767856;
        new Uint8Array(memory.buffer).set(new Uint8Array(testInput.buffer), inputPtr);
        
        console.log('   Executando Layer 0...');
        const startTime = Date.now();
        instance.exports.run_layer(0);
        const execTime = Date.now() - startTime;
        
        // Ler saída
        const layer0OutPtr = layer0Params[4];
        const output = new Int8Array(memory.buffer, layer0OutPtr, 16 * 112 * 112);
        
        const outputStats = {
            min: Math.min(...Array.from(output).slice(0, 1000)),
            max: Math.max(...Array.from(output).slice(0, 1000)),
            avg: Array.from(output).slice(0, 1000).reduce((a, b) => a + b, 0) / 1000,
            unique: new Set(Array.from(output).slice(0, 1000)).size,
            sample: Array.from(output).slice(0, 20),
            executionTimeMs: execTime
        };
        
        diagnosticResults.syntheticTest = outputStats;
        
        console.log('\n   Saída Layer 0 (primeiros 1000 valores):');
        console.log(`      Min: ${outputStats.min}`);
        console.log(`      Max: ${outputStats.max}`);
        console.log(`      Avg: ${outputStats.avg.toFixed(2)}`);
        console.log(`      Valores únicos: ${outputStats.unique}`);
        console.log(`      Tempo de execução: ${execTime}ms`);
        console.log(`      Sample: [${outputStats.sample.join(', ')}]`);
        
        if (outputStats.unique === 1) {
            const msg = '❌ CRÍTICO: Saída tem apenas 1 valor único! A convolução não está funcionando.';
            console.log('\n' + msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'critical', category: 'output', message: msg });
        } else if (outputStats.min === outputStats.max) {
            const msg = '❌ CRÍTICO: Todos os valores são iguais!';
            console.log('\n' + msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'critical', category: 'output', message: msg });
        } else {
            const msg = '✅ Layer 0 está produzindo saída variada';
            console.log('\n' + msg + '\n');
            diagnosticResults.diagnosis.push({ severity: 'ok', category: 'output', message: msg });
        }
        
        // Salvar resultados
        console.log('═══════════════════════════════════════════════════════\n');
        console.log('💾 Salvando resultados do diagnóstico...\n');
        
        fs.writeFileSync('diagnostic_weights.json', JSON.stringify(diagnosticResults, null, 2));
        console.log('✅ Salvo: diagnostic_weights.json');
        
        // Salvar binários para análise detalhada
        fs.writeFileSync('weights_layer0_raw.bin', Buffer.from(weights));
        console.log('✅ Salvo: weights_layer0_raw.bin (primeiros 100 pesos)');
        
        fs.writeFileSync('output_layer0_synthetic.bin', Buffer.from(output.slice(0, 1000)));
        console.log('✅ Salvo: output_layer0_synthetic.bin (primeiros 1000 valores de saída)');
        
        // Resumo final
        console.log('\n═══════════════════════════════════════════════════════');
        console.log('📊 RESUMO DO DIAGNÓSTICO:\n');
        
        const errors = diagnosticResults.diagnosis.filter(d => d.severity === 'error' || d.severity === 'critical');
        const oks = diagnosticResults.diagnosis.filter(d => d.severity === 'ok');
        
        console.log(`   ✅ Testes OK: ${oks.length}`);
        console.log(`   ❌ Problemas encontrados: ${errors.length}\n`);
        
        if (errors.length > 0) {
            console.log('   Problemas detectados:');
            errors.forEach((err, i) => {
                console.log(`   ${i + 1}. [${err.category}] ${err.message}`);
            });
        } else {
            console.log('   🎉 Todos os testes passaram!\n');
            console.log('   💡 Se a classificação está ruim, o problema pode ser:');
            console.log('      - Normalização/pré-processamento da imagem');
            console.log('      - Ordem dos canais (RGB vs BGR)');
            console.log('      - Escala de quantização diferente do esperado');
        }
        
        console.log('\n═══════════════════════════════════════════════════════\n');
        
    } catch (error) {
        console.error('❌ ERRO:', error.message);
        console.error(error.stack);
        
        diagnosticResults.diagnosis.push({
            severity: 'critical',
            category: 'execution',
            message: error.message,
            stack: error.stack
        });
        
        fs.writeFileSync('diagnostic_weights_error.json', JSON.stringify(diagnosticResults, null, 2));
        console.log('\n💾 Erro salvo em: diagnostic_weights_error.json\n');
    }
})();