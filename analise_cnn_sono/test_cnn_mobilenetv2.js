import fs from "fs";

const IMAGE_PATH = "dirigir-com-sono-uint8.raw"; // rgb565 raw: 128*128*2 = 32768 bytes
const WASM_PATH  = "main.wasm";

const LABELS = ["acordado", "com_sono"];

const INPUT_PTR        = 507248; // SLOT0
const FORMAT_FLAG_ADDR = 0;      // endereço do byte de formato
const FORMAT_RGB565    = 65;     // sentinela RGB565
const NUM_CLASSES      = 2;
const RGB565_BYTES     = 128 * 128 * 2; // 32768

(async () => {

  // ===== Carregar WASM =====
  const wasmBuffer = fs.readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(wasmBuffer, {
    env: {
      log:   (v) => console.log("[WAT log]", v),
      logf:  (v) => console.log("[WAT logf]", v),
      log64: (v) => console.log("[WAT log64]", v),
    }
  });

  const memory = instance.exports.memory;
  const memU8  = new Uint8Array(memory.buffer);

  // ===== Carregar imagem RGB565 =====
  const image = fs.readFileSync(IMAGE_PATH);

  if (image.length !== RGB565_BYTES) {
    console.error(`❌ Tamanho incorreto: ${image.length} bytes, esperado ${RGB565_BYTES} (128×128×2 RGB565)`);
    process.exit(1);
  }

  memU8.set(image, INPUT_PTR);
  memU8[FORMAT_FLAG_ADDR] = FORMAT_RGB565;

  // ===== Executar rede =====
  instance.exports.run_mobilenetv2();

  // ===== Ler saída =====
  const resultPtr = instance.exports.get_result_ptr();
  const probs = new Uint8Array(memory.buffer, resultPtr, NUM_CLASSES);

  console.log("\n===== RESULTADO =====\n");
  for (let i = 0; i < NUM_CLASSES; i++) {
    const q = probs[i];
    console.log(`${LABELS[i]} => ${((q / 256) * 100).toFixed(2)}% (q=${q})`);
  }

})();
