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

    console.log('🔍 DEBUG ESPECÍFICO DA LAYER 62 (MEAN)\n');

    const wasmBuffer = fs.readFileSync("main.wasm");
    const wasmModule = await WebAssembly.instantiate(wasmBuffer);
    const instance = wasmModule.instance;
    const memory = instance.exports.memory;

    // ============================================================
    // Carregar imagem branca
    // ============================================================

    const whiteImage = Buffer.alloc(224 * 224 * 3, 255);
    const rgb565 = rgb888ToRgb565(whiteImage, 224, 224);
    const rgb565Bytes = new Uint8Array(rgb565.buffer);

    const inputPtr = 1767856;
    new Uint8Array(memory.buffer).set(rgb565Bytes, inputPtr);

    console.log('✅ Imagem branca carregada\n');

    // ============================================================
    // Executar layers 0-61
    // ============================================================

    console.log('⚙️  Executando layers 0-61...\n');
    for (let i = 0; i <= 61; i++) {
        instance.exports.run_layer(i);
    }

    // ============================================================
    // Ler LayerParam 62 corretamente
    // ============================================================

    const LAYERPARAM_BASE = 1760576;
    const LAYERPARAM_SIZE = 112; // bytes

    const layer62Base = LAYERPARAM_BASE + 62 * LAYERPARAM_SIZE;
    const layer62 = new Int32Array(memory.buffer, layer62Base, LAYERPARAM_SIZE / 4);

    const opType = layer62[0];
    const inPtr  = layer62[3];
    const outPtr = layer62[4];
    const inH    = layer62[5];
    const inW    = layer62[6];
    const cin    = layer62[7];
    const zX     = layer62[23];
    const zY     = layer62[25];

    console.log('═══════════════════════════════════════════════════════');
    console.log('📋 LAYER 62 (MEAN) - PARÂMETROS:\n');

    console.log(`   op_type: ${opType} (esperado: 5 = MEAN)`);
    console.log(`   in_ptr: ${inPtr}`);
    console.log(`   out_ptr: ${outPtr}`);
    console.log(`   cin: ${cin}`);
    console.log(`   in_h: ${inH}`);
    console.log(`   in_w: ${inW}`);
    console.log(`   zX: ${zX}`);
    console.log(`   zY: ${zY}\n`);

    // Dump inicial da struct (debug estrutural)
    console.log('🧪 RAW LayerParam (primeiros 12 campos):');
    for (let i = 0; i < 12; i++) {
        console.log(`   [${i}] = ${layer62[i]}`);
    }

    // ============================================================
    // INPUT DEBUG
    // ============================================================

    console.log('\n📊 INPUT DA MEAN (Layer 61 output):\n');

    const spatialSize = inH * inW;
    const totalInputSize = spatialSize * cin;

    if (totalInputSize <= 0) {
        console.log('❌ ERRO: spatial_size inválido!');
        console.log(`   inH=${inH}, inW=${inW}, cin=${cin}`);
    } else {

        const inputTensor = new Int8Array(memory.buffer, inPtr, totalInputSize);

        let min = 127;
        let max = -128;
        let sum = 0;
        const unique = new Set();

        for (let i = 0; i < inputTensor.length; i++) {
            const v = inputTensor[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            unique.add(v);
        }

        const avg = sum / inputTensor.length;

        console.log(`   Dimensões: ${inH}x${inW}x${cin}`);
        console.log(`   Total pixels: ${inputTensor.length}`);
        console.log(`   Min: ${min}`);
        console.log(`   Max: ${max}`);
        console.log(`   Avg: ${avg.toFixed(4)}`);
        console.log(`   Valores únicos: ${unique.size}`);

        // Primeiro canal completo
        const firstChannel = [];
        for (let i = 0; i < spatialSize; i++) {
            firstChannel.push(inputTensor[i * cin]);
        }

        console.log(`   Primeiro canal (${spatialSize} pixels):`);
        console.log(firstChannel.slice(0, 49));
    }

    // ============================================================
    // Executar MEAN
    // ============================================================

    console.log('\n⚙️  Executando Layer 62 (MEAN)...\n');
    instance.exports.run_layer(62);

    // ============================================================
    // OUTPUT DEBUG
    // ============================================================

    console.log('📤 OUTPUT DA MEAN:\n');

    const outputTensor = new Int8Array(memory.buffer, outPtr, cin);

    let minO = 127;
    let maxO = -128;
    let sumO = 0;
    const uniqueO = new Set();

    for (let i = 0; i < outputTensor.length; i++) {
        const v = outputTensor[i];
        if (v < minO) minO = v;
        if (v > maxO) maxO = v;
        sumO += v;
        uniqueO.add(v);
    }

    const avgO = sumO / outputTensor.length;

    console.log(`   Min: ${minO}`);
    console.log(`   Max: ${maxO}`);
    console.log(`   Avg: ${avgO.toFixed(4)}`);
    console.log(`   Valores únicos: ${uniqueO.size}`);
    console.log(`   Sample [0-19]:`);
    console.log(Array.from(outputTensor.slice(0, 20)));

    console.log('\n═══════════════════════════════════════════════════════\n');

    // ============================================================
    // Salvar JSON
    // ============================================================

    const debugData = {
        layer62Params: {
            opType,
            inPtr,
            outPtr,
            inH,
            inW,
            cin,
            zX,
            zY
        },
        outputSample: Array.from(outputTensor.slice(0, 100))
    };

    fs.writeFileSync('debug_mean.json', JSON.stringify(debugData, null, 2));
    console.log('💾 Debug salvo em: debug_mean.json\n');

})().catch(err => {
    console.error('❌ ERRO:', err.message);
    console.error(err.stack);
});
