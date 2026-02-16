const fs = require("fs");

const IMAGE_PATH = "aviao_uint8.raw";
const WASM_PATH = "main.wasm";

const INPUT_PTR = 1768080;
const PARAMS_BASE = 1760576;
const LP_SIZE = 116;
const NUM_LAYERS = 67;
const NUM_CLASSES = 1000;

const FIELD = {
    op_type: 0,
    out_ptr: 4,
    cout: 8
};

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

(async () => {

  console.log("🚀 Executando MobileNetV2\n");

  const wasmBuffer = fs.readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(wasmBuffer);

  const memory = instance.exports.memory;
  const memU8 = new Uint8Array(memory.buffer);

  console.log("✅ WASM carregado");
  console.log("Memory size:", memory.buffer.byteLength, "bytes");
  console.log("Memory pages:", memory.buffer.byteLength / 65536);

  // ==========================
  // Inserir imagem
  // ==========================
  const image = fs.readFileSync(IMAGE_PATH);

  if (image.length !== 224 * 224 * 3) {
    console.error("❌ Tamanho incorreto da imagem!");
    process.exit(1);
  }

  memU8.set(image, INPUT_PTR);
  console.log("📷 Imagem inserida\n");

  // ==========================
  // Executar rede
  // ==========================
  console.log("⚙️ Executando rede...");
  const start = Date.now();

  instance.exports.run_mobilenetv2();

  for (let i = 0; i < 5; i++) {
    const base = PARAMS_BASE + i * LP_SIZE;
    const view = new Int32Array(memory.buffer, base, 5);
    console.log("Layer", i, "op_type:", view[0]);
  }

  const end = Date.now();
  console.log(`⏱ Tempo: ${end - start} ms\n`);

  // ==========================
  // Descobrir ponteiro da saída final - COM DEBUG
  // ==========================
  const lastBase = PARAMS_BASE + (NUM_LAYERS - 1) * LP_SIZE;
  console.log("\n🔍 DEBUG - Última Layer:");
  console.log("lastBase:", lastBase);
  
  const lastView = new Int32Array(memory.buffer, lastBase, 29);
  
  console.log("\nPrimeiros 10 valores da última layer:");
  for (let i = 0; i < 10; i++) {
    console.log(`  [${i}] = ${lastView[i]}`);
  }
  
  console.log("\nTentando diferentes índices para out_ptr:");
  console.log("  lastView[1] =", lastView[1]);
  console.log("  lastView[4] =", lastView[4]);
  console.log("  lastView[FIELD.out_ptr] = lastView[4] =", lastView[FIELD.out_ptr]);
  console.log("  lastView[FIELD.out_ptr/4] = lastView[1] =", lastView[FIELD.out_ptr/4]);
  
  // Testar qual índice funciona
  let OUTPUT_PTR;
  
  // Tentar índice 1
  if (lastView[1] > 0 && lastView[1] + NUM_CLASSES <= memory.buffer.byteLength) {
    console.log("\n✅ Usando lastView[1]");
    OUTPUT_PTR = lastView[1];
  }
  // Tentar índice 4
  else if (lastView[4] > 0 && lastView[4] + NUM_CLASSES <= memory.buffer.byteLength) {
    console.log("\n✅ Usando lastView[4]");
    OUTPUT_PTR = lastView[4];
  }
  else {
    console.log("\n❌ Nenhum índice válido encontrado!");
    console.log("\n🔍 Procurando em todas as últimas layers:");
    
    for (let i = NUM_LAYERS - 5; i < NUM_LAYERS; i++) {
      const base = PARAMS_BASE + i * LP_SIZE;
      const view = new Int32Array(memory.buffer, base, 29);
      console.log(`\nLayer ${i}:`);
      console.log(`  op_type [0]: ${view[0]}`);
      
      for (let j = 0; j < 10; j++) {
        const ptr = view[j];
        if (ptr > 0 && ptr + NUM_CLASSES <= memory.buffer.byteLength) {
          console.log(`  [${j}]: ${ptr} ✅ (válido para ${NUM_CLASSES} bytes)`);
        } else if (ptr > 0) {
          console.log(`  [${j}]: ${ptr} ❌ (fora dos limites)`);
        }
      }
    }
    process.exit(1);
  }
  
  console.log("\nOUTPUT_PTR:", OUTPUT_PTR);
  console.log("Memory size:", memory.buffer.byteLength);
  console.log("Espaço disponível:", memory.buffer.byteLength - OUTPUT_PTR, "bytes");
  console.log("Necessário:", NUM_CLASSES, "bytes");

  const logits = new Int8Array(memory.buffer, OUTPUT_PTR, NUM_CLASSES);

  console.log("\n📊 Logits:");
  console.log("Valores únicos nos logits:", new Set(logits).size);
  console.log("Primeiros 20 logits:", Array.from(logits.slice(0, 20)));

  // ==========================
  // Softmax
  // ==========================
  const logitsFloat = Array.from(logits).map(v => v * 0.00390625);
  const probs = softmax(logitsFloat);

  const results = probs
    .map((p, i) => ({ index: i, prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);

  console.log("\n🏆 TOP-5 RESULTADOS\n");

  results.forEach((r, i) => {
    console.log(
      `${i + 1}. ${(r.prob * 100).toFixed(2)}% → classe ${r.index}`
    );
  });

})();