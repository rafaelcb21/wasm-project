const fs = require('fs');

(async () => {
    console.log('🔍 INSPEÇÃO COMPLETA DOS PARÂMETROS - LAYER 62\n');
    
    const wasmBuffer = fs.readFileSync("main.wasm");
    const wasmModule = await WebAssembly.instantiate(wasmBuffer);
    const instance = wasmModule.instance;
    const memory = instance.exports.memory;
    
    // Ler TODOS os 28 parâmetros da Layer 62
    const layer62ParamsBase = 1760576 + 62 * 112;
    const allParams = new Int32Array(memory.buffer, layer62ParamsBase, 28);
    
    console.log('═══════════════════════════════════════════════════════');
    console.log('📋 TODOS OS 28 PARÂMETROS DA LAYER 62:\n');
    
    const fieldNames = [
        'op_type',      // 0
        'relu',         // 1
        'pad_value',    // 2
        'in_ptr',       // 3
        'out_ptr',      // 4
        'out_h',        // 5
        'out_w',        // 6
        'cin',          // 7
        'cout',         // 8
        'kh',           // 9
        'kw',           // 10
        'stride_h',     // 11
        'stride_w',     // 12
        'dilation_h',   // 13
        'dilation_w',   // 14
        'pad_t',        // 15
        'in_h',         // 16
        'in_w',         // 17
        'pad_b',        // 18
        'wptr',         // 19
        'bias_ptr',     // 20
        'mul',          // 21
        'shift',        // 22
        'zx',           // 23
        'zw',           // 24
        'zy',           // 25
        'out_h_actual', // 26
        'out_w_actual'  // 27
    ];
    
    for (let i = 0; i < 28; i++) {
        const name = fieldNames[i] || `field_${i}`;
        const value = allParams[i];
        const highlight = (value === 0 && i >= 16 && i <= 22) ? '❌' : '  ';
        console.log(`   [${i.toString().padStart(2)}] ${highlight} ${name.padEnd(15)} = ${value}`);
    }
    
    console.log('\n═══════════════════════════════════════════════════════\n');
    
    // Agora vamos verificar a Layer 61 (que alimenta a Layer 62)
    const layer61ParamsBase = 1760576 + 61 * 112;
    const layer61Params = new Int32Array(memory.buffer, layer61ParamsBase, 28);
    
    console.log('📋 PARÂMETROS DA LAYER 61 (CONV antes do MEAN):\n');
    
    const relevantFields61 = {
        'op_type': 0,
        'out_ptr': 4,
        'out_h': 26,
        'out_w': 27,
        'cout': 8
    };
    
    for (const [name, idx] of Object.entries(relevantFields61)) {
        console.log(`   ${name.padEnd(10)} = ${layer61Params[idx]}`);
    }
    
    console.log('\n═══════════════════════════════════════════════════════\n');
    console.log('💡 ANÁLISE:\n');
    
    if (allParams[16] === 0 || allParams[17] === 0) {
        console.log('❌ PROBLEMA: in_h e in_w da Layer 62 são ZERO!\n');
        console.log('   Isso significa que os parâmetros não foram configurados');
        console.log('   corretamente quando você gerou o arquivo de parâmetros.\n');
        console.log('   A Layer 62 (MEAN) deveria ter:');
        console.log(`      in_h = ${layer61Params[26]} (out_h da Layer 61)`);
        console.log(`      in_w = ${layer61Params[27]} (out_w da Layer 61)`);
        console.log(`      in_ptr = ${layer61Params[4]} (out_ptr da Layer 61)\n`);
        console.log('   Mas tem:');
        console.log(`      in_h = ${allParams[16]}`);
        console.log(`      in_w = ${allParams[17]}`);
        console.log(`      in_ptr = ${allParams[3]}\n`);
    }
    
    if (allParams[21] === 0 || allParams[22] === 0) {
        console.log('❌ PROBLEMA: mul e shift da Layer 62 são ZERO!\n');
        console.log('   Os parâmetros de quantização não foram carregados.\n');
    }
    
    console.log('═══════════════════════════════════════════════════════\n');
    
    // Salvar para análise
    const debugData = {
        layer62_all_params: Array.from(allParams),
        layer61_output_info: {
            out_ptr: layer61Params[4],
            out_h: layer61Params[26],
            out_w: layer61Params[27],
            cout: layer61Params[8]
        }
    };
    
    fs.writeFileSync('params_inspection.json', JSON.stringify(debugData, null, 2));
    console.log('💾 Inspeção salva em: params_inspection.json\n');
    
})().catch(err => {
    console.error('❌ ERRO:', err.message);
});