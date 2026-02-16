const fs = require('fs');

/**
 * Script para verificar se os layers estão sendo executados
 */

(async () => {
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║         VERIFICAÇÃO DE EXECUÇÃO DOS LAYERS                ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');
    
    try {
        // Carregar WASM
        console.log('📦 Carregando WASM...');
        const wasmBuffer = fs.readFileSync("main.wasm");
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        const instance = wasmModule.instance;
        const memory = instance.exports.memory;
        
        console.log('✅ WASM carregado\n');
        
        // Carregar imagem
        console.log('📷 Carregando imagem...');
        const rawImage = fs.readFileSync("aviao_uint8.raw");
        
        if (rawImage.length !== 224 * 224 * 3) {
            throw new Error('Tamanho incorreto!');
        }
               
        const inputPtr = 1767856;
        const memoryView = new Uint8Array(memory.buffer);
        memoryView.set(rawImage, inputPtr);
        
        console.log('✅ Imagem carregada\n');
        
        // Verificar funções exportadas
        console.log('🔍 Funções exportadas disponíveis:');
        const exports = Object.keys(instance.exports);
        const layerFunctions = exports.filter(e => 
            e.includes('layer') || 
            e.includes('conv') || 
            e.includes('depthwise') ||
            e.includes('fully') ||
            e.includes('softmax') ||
            e.includes('mean') ||
            e.includes('add') ||
            e.includes('quantize')
        );
        
        layerFunctions.forEach(fn => {
            console.log(`   • ${fn}`);
        });
        console.log();
        
        // Testar execução layer por layer
        console.log('🧪 Testando execução individual de layers...\n');
        
        // Verificar se run_layer existe
        if (instance.exports.run_layer) {
            console.log('✅ Função run_layer encontrada\n');
            
            console.log('Executando layers 64, 65 e 66 individualmente...\n');
            
            // Primeiro, executar layers 0-63 normalmente
            console.log('📊 Executando layers 0-63...');
            for (let i = 0; i < 64; i++) {
                instance.exports.run_layer(i);
            }
            console.log('✅ Layers 0-63 executados\n');
            
            // Capturar saída após layer 64 (Mean)
            console.log('📊 Executando Layer 64 (Mean)...');
            instance.exports.run_layer(64);
            const resultPtr = instance.exports.get_result_ptr();
            const afterMean = new Int8Array(memory.buffer, resultPtr, 1280);
            const meanStats = {
                min: Math.min(...afterMean),
                max: Math.max(...afterMean),
                avg: Array.from(afterMean).reduce((a,b) => a+b, 0) / afterMean.length
            };
            console.log(`   Saída (1280 features):`);
            console.log(`   Range: [${meanStats.min}, ${meanStats.max}]`);
            console.log(`   Média: ${meanStats.avg.toFixed(2)}\n`);
            
            // Capturar saída após layer 65 (FC)
            console.log('📊 Executando Layer 65 (Fully Connected)...');
            instance.exports.run_layer(65);
            const afterFC = new Int8Array(memory.buffer, resultPtr, 1000);
            const fcStats = {
                min: Math.min(...afterFC),
                max: Math.max(...afterFC),
                avg: Array.from(afterFC).reduce((a,b) => a+b, 0) / afterFC.length,
                saturated127: Array.from(afterFC).filter(v => v === 127).length,
                saturatedMinus128: Array.from(afterFC).filter(v => v === -128).length
            };
            console.log(`   Saída (1000 logits):`);
            console.log(`   Range: [${fcStats.min}, ${fcStats.max}]`);
            console.log(`   Média: ${fcStats.avg.toFixed(2)}`);
            console.log(`   Saturação +127: ${fcStats.saturated127}`);
            console.log(`   Saturação -128: ${fcStats.saturatedMinus128}\n`);
            
            // Salvar logits antes do softmax
            fs.writeFileSync('logits_before_softmax.txt', 
                Array.from(afterFC).map((v, i) => `${i}\t${v}`).join('\n')
            );
            console.log('💾 Salvos: logits_before_softmax.txt\n');
            
            // Capturar saída após layer 66 (Softmax)
            console.log('📊 Executando Layer 66 (Softmax)...');
            instance.exports.run_layer(66);
            const afterSoftmax = new Int8Array(memory.buffer, resultPtr, 1000);
            const softmaxStats = {
                min: Math.min(...afterSoftmax),
                max: Math.max(...afterSoftmax),
                avg: Array.from(afterSoftmax).reduce((a,b) => a+b, 0) / afterSoftmax.length,
                saturated127: Array.from(afterSoftmax).filter(v => v === 127).length,
                saturatedMinus128: Array.from(afterSoftmax).filter(v => v === -128).length,
                unique: new Set(afterSoftmax).size
            };
            console.log(`   Saída (1000 probabilidades):`);
            console.log(`   Range: [${softmaxStats.min}, ${softmaxStats.max}]`);
            console.log(`   Média: ${softmaxStats.avg.toFixed(2)}`);
            console.log(`   Valores únicos: ${softmaxStats.unique}`);
            console.log(`   Saturação +127: ${softmaxStats.saturated127}`);
            console.log(`   Saturação -128: ${softmaxStats.saturatedMinus128}\n`);
            
            // Salvar probabilidades após softmax
            fs.writeFileSync('probs_after_softmax.txt', 
                Array.from(afterSoftmax).map((v, i) => `${i}\t${v}`).join('\n')
            );
            console.log('💾 Salvos: probs_after_softmax.txt\n');
            
            // Análise comparativa
            console.log('╔════════════════════════════════════════════════════════════╗');
            console.log('║                   ANÁLISE COMPARATIVA                      ║');
            console.log('╚════════════════════════════════════════════════════════════╝\n');
            
            console.log('📊 FC → Softmax:');
            console.log(`   Saturação +127: ${fcStats.saturated127} → ${softmaxStats.saturated127}`);
            console.log(`   Saturação -128: ${fcStats.saturatedMinus128} → ${softmaxStats.saturatedMinus128}`);
            console.log(`   Valores únicos: ? → ${softmaxStats.unique}\n`);
            
            if (fcStats.saturated127 > 100) {
                console.log('🔴 PROBLEMA: Layer 65 (FC) já está saturando!');
                console.log('   Causa: Multiplicadores Q31 muito altos ou bias incorretos\n');
            }
            
            if (softmaxStats.saturated127 > 100) {
                console.log('🔴 PROBLEMA: Layer 66 (Softmax) aumentou a saturação!');
                console.log('   Causa: Parâmetros de quantização do Softmax incorretos\n');
            }
            
            if (softmaxStats.unique < 100) {
                console.log('🔴 PROBLEMA: Softmax produziu poucos valores únicos!');
                console.log('   Causa: Quantização muito agressiva ou implementação errada\n');
            }
            
        } else {
            console.log('❌ Função run_layer NÃO encontrada');
            console.log('   Executando rede completa...\n');
            
            const startTime = Date.now();
            instance.exports.run_mobilenetv2();
            const endTime = Date.now();
            
            console.log(`✅ Rede completa em ${endTime - startTime}ms\n`);
            
            const resultPtr = instance.exports.get_result_ptr();
            const output = new Int8Array(memory.buffer, resultPtr, 1000);
            
            const stats = {
                min: Math.min(...output),
                max: Math.max(...output),
                avg: Array.from(output).reduce((a,b) => a+b, 0) / output.length,
                saturated127: Array.from(output).filter(v => v === 127).length,
                saturatedMinus128: Array.from(output).filter(v => v === -128).length,
                unique: new Set(output).size
            };
            
            console.log('📊 Saída final:');
            console.log(`   Range: [${stats.min}, ${stats.max}]`);
            console.log(`   Média: ${stats.avg.toFixed(2)}`);
            console.log(`   Valores únicos: ${stats.unique}`);
            console.log(`   Saturação +127: ${stats.saturated127}`);
            console.log(`   Saturação -128: ${stats.saturatedMinus128}\n`);
        }
        
        console.log('═'.repeat(70));
        console.log('VERIFICAÇÃO CONCLUÍDA');
        console.log('═'.repeat(70) + '\n');
        
    } catch (error) {
        console.error('\n❌ ERRO:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();
