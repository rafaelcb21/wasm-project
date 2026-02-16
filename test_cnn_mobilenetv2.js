const fs = require("fs");

const IMAGE_PATH = "aviao_uint8.raw";
const WASM_PATH = "main.wasm";

const INPUT_PTR = 1800512;
const PARAMS_BASE = 1792736;
const LP_SIZE = 116;
const NUM_LAYERS = 67;
const NUM_CLASSES = 1000;

const FIELD = {
    op_type: 0,
    in_ptr: 3,
    out_ptr: 4,
    cout: 8,
    wptr: 19,
    bias_ptr: 20,
    mul_ptr: 21,
    shift_ptr: 22
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
  const { instance } = await WebAssembly.instantiate(wasmBuffer, {
        env: {
            log: (value) => console.log("LOG:", value),
        }
    });

  const memory = instance.exports.memory;
  const memU8 = new Uint8Array(memory.buffer);

  console.log("✅ WASM carregado");

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

  const end = Date.now();
  console.log(`⏱ Tempo: ${end - start} ms\n`);

  // Após executar a rede
  const resultPtr = instance.exports.get_result_ptr();
  const logits = new Int8Array(memory.buffer, resultPtr, NUM_CLASSES);

  console.log("\n📊 Análise dos Logits:");
  console.log("Min:", Math.min(...logits));
  console.log("Max:", Math.max(...logits));
  console.log("Valores únicos:", new Set(logits).size);
  console.log("Primeiros 20:", Array.from(logits.slice(0, 20)));

  // Dequantizar para float (assumindo scale 0.00390625, zp 0)
  const scale = 0.00390625; // 1/256
  const zeroPoint = 0;
  const logitsFloat = Array.from(logits).map(v => (v - zeroPoint) * scale);

  // Aplicar softmax em float
  const probs = softmax(logitsFloat);

  // Top-5
  const top5 = probs
    .map((p, i) => ({ index: i, prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);

  console.log("\n🏆 TOP-5 RESULTADOS:\n");
  top5.forEach((r, i) => {
    console.log(`${i + 1}. ${(r.prob * 100).toFixed(2)}% → classe ${r.index}`);
  });

})();
