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
    console.log('🔍 DEBUG DETALHADO - RASTREANDO A ENTRADA\n');
    
    const wasmBuffer = fs.readFileSync("main.wasm");
    const wasmModule = await WebAssembly.instantiate(wasmBuffer);
    const instance = wasmModule.instance;
    const memory = instance.exports.memory;
    
    // Criar duas imagens MUITO diferentes
    const blackImage = Buffer.alloc(224 * 224 * 3, 0);
    const whiteImage = Buffer.alloc(224 * 224 * 3, 255);
    
    const tests = [
        { name: 'PRETA', buffer: blackImage, expectedRGB565: 0x0000 },
        { name: 'BRANCA', buffer: whiteImage, expectedRGB565: 0xFFFF }
    ];
    
    for (const test of tests) {
        console.log(`═══════════════════════════════════════════════════════`);
        console.log(`🖼️  Teste: ${test.name}\n`);
        
        // Converter para RGB565
        const rgb565 = rgb888ToRgb565(test.buffer, 224, 224);
        const rgb565Bytes = new Uint8Array(rgb565.buffer);
        
        console.log(`   📊 Entrada RGB565:`);
        console.log(`      Primeiro pixel: 0x${rgb565[0].toString(16).padStart(4, '0')}`);
        console.log(`      Esperado: 0x${test.expectedRGB565.toString(16).padStart(4, '0')}`);
        console.log(`      Match: ${rgb565[0] === test.expectedRGB565 ? '✅' : '❌'}\n`);
        
        // Carregar na memória
        const inputPtr = 1767856;
        new Uint8Array(memory.buffer).set(rgb565Bytes, inputPtr);
        
        // Verificar se foi escrito
        const readback = new Uint16Array(memory.buffer, inputPtr, 10);
        console.log(`   📥 Verificação na Memória WASM:`);
        console.log(`      Primeiros 10 pixels: [${Array.from(readback).map(v => '0x' + v.toString(16).padStart(4, '0')).join(', ')}]`);
        console.log(`      Todos iguais: ${new Set(Array.from(readback)).size === 1 ? '✅' : '❌'}\n`);
        
        // Executar APENAS Layer 0
        console.log(`   ⚙️  Executando Layer 0...`);
        instance.exports.run_layer(0);
        
        // Ler saída da Layer 0
        const layer0ParamsBase = 1760576;
        const layer0Params = new Int32Array(memory.buffer, layer0ParamsBase, 28);
        const layer0OutPtr = layer0Params[4];
        
        const output = new Int8Array(memory.buffer, layer0OutPtr, 1000);
        
        const stats = {
            min: Math.min(...Array.from(output)),
            max: Math.max(...Array.from(output)),
            avg: Array.from(output).reduce((a, b) => a + b, 0) / 1000,
            unique: new Set(Array.from(output)).size,
            sample: Array.from(output).slice(0, 20)
        };
        
        console.log(`   📤 Saída Layer 0:`);
        console.log(`      Min: ${stats.min}, Max: ${stats.max}, Avg: ${stats.avg.toFixed(2)}`);
        console.log(`      Valores únicos: ${stats.unique}`);
        console.log(`      Sample: [${stats.sample.join(', ')}]\n`);
    }
    
    console.log(`═══════════════════════════════════════════════════════\n`);
    console.log(`🔍 ANÁLISE:\n`);
    
    console.log(`   Se as saídas da Layer 0 são IDÊNTICAS para imagens`);
    console.log(`   completamente diferentes (preta vs branca), então:\n`);
    console.log(`   ❌ A Layer 0 NÃO está lendo a entrada corretamente`);
    console.log(`   ❌ O endereço de entrada pode estar errado`);
    console.log(`   ❌ A função conv2d_layer0 pode ter um bug\n`);
    
    console.log(`═══════════════════════════════════════════════════════\n`);
    
})().catch(err => {
    console.error('❌ ERRO:', err.message);
    console.error(err.stack);
});