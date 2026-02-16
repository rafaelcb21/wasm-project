const fs = require('fs');
const sharp = require('sharp');

const INPUT_PTR = 1768080;   // ajuste se necessário
const PARAMS_BASE = 1760576;
const LP_SIZE = 112;
const NUM_LAYERS = 67;

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a,b)=>a+b,0);
    return exps.map(v => v / sum);
}

function loadImageNetLabels(filename = 'imagenet_labels.txt') {
    try {
        const content = fs.readFileSync(filename, 'utf8');
        return content.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    } catch {
        return Array.from({length:1000},(_,i)=>`Class_${i}`);
    }
}

(async () => {

    console.log("🚀 Executando MobileNetV2\n");

    // 1. Carregar WASM
    const wasmBuffer = fs.readFileSync("main.wasm");
    const wasmModule = await WebAssembly.instantiate(wasmBuffer);
    const instance = wasmModule.instance;
    const memory = instance.exports.memory;
    const memU8 = new Uint8Array(memory.buffer);

    console.log("✅ WASM carregado");

    // 2. Carregar imagem do avião
    console.log("📷 Processando imagem...");

    const rgbBuffer = await sharp("aviao.jpg")
        .resize(224,224)
        .removeAlpha()
        .raw()
        .toBuffer();

    if (rgbBuffer.length !== 224*224*3) {
        throw new Error("Imagem não está em RGB888 224x224x3");
    }

    console.log("   Tamanho:", rgbBuffer.length, "bytes");

    // 3. Copiar para memória WASM
    memU8.set(rgbBuffer, INPUT_PTR);

    console.log("✅ Imagem inserida na memória\n");

    // 4. Executar todas as camadas
    console.log("⚙️  Executando rede...");
    const t0 = Date.now();

    for (let i = 0; i < NUM_LAYERS; i++) {
        instance.exports.run_layer(i);
    }

    const t1 = Date.now();
    console.log("⏱ Tempo:", t1 - t0, "ms\n");

    // 5. Ler saída da FC (antes do Softmax quantizado)
    const layer64Base = PARAMS_BASE + 64 * LP_SIZE;
    const layerParams = new Int32Array(memory.buffer, layer64Base, 28);
    const logitsPtr = layerParams[3];  // in_ptr do softmax

    const logits = new Int8Array(memory.buffer, logitsPtr, 1000);

    // 6. Converter para float usando escala final
    const scaleOut = 0.00390625; // do output_0
    const logitsFloat = Array.from(logits).map(v => v * scaleOut);

    const probs = softmax(logitsFloat);

    // 7. Ranking
    const labels = loadImageNetLabels();

    const results = probs.map((p,i)=>({
        index: i,
        prob: p
    })).sort((a,b)=>b.prob - a.prob);

    console.log("═══════════════════════════════");
    console.log("🏆 TOP-5 RESULTADOS");
    console.log("═══════════════════════════════\n");

    for (let i=0;i<5;i++){
        const r = results[i];
        console.log(
            `${i+1}. ${(r.prob*100).toFixed(2)}%  →  ${labels[r.index]}`
        );
    }

    console.log("\n✨ Concluído");

})();
