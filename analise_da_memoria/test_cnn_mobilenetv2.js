import fs from "fs";

const IMAGE_PATH = "aviao_uint8.raw";
const WASM_PATH = "main.wasm";
const LABELS_PATH = "imagenet_labels.txt";

const INPUT_PTR = 1800512;
const NUM_CLASSES = 1000;

(async () => {

  // ===== Carregar labels =====
  const labels = fs
    .readFileSync(LABELS_PATH, "utf8")
    .split("\n")
    .map(l => l.trim())
    .filter(l => l.length > 0);

  if (labels.length !== NUM_CLASSES) {
    console.error("❌ Arquivo imagenet_labels.txt não tem 1000 linhas!");
    process.exit(1);
  }

  // ===== Carregar WASM =====
  const wasmBuffer = fs.readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(wasmBuffer, {
    env: {
      log: (value) => console.log("LOG:", value),
      logf: (value) => console.log("LOGF:", value),
      log64: (value) => console.log("LOG64:", value),
    }
  });

  const memory = instance.exports.memory;
  const memU8 = new Uint8Array(memory.buffer);

  // ===== Carregar imagem =====
  const image = fs.readFileSync(IMAGE_PATH);

  if (image.length !== 224 * 224 * 3) {
    console.error("❌ Tamanho incorreto da imagem!");
    process.exit(1);
  }

  memU8.set(image, INPUT_PTR);

  // ===== Executar rede =====
  instance.exports.run_mobilenetv2();

  // ===== Ler saída =====
  const resultPtr = instance.exports.get_result_ptr();
  const probs = new Uint8Array(memory.buffer, resultPtr, NUM_CLASSES);

  const classes = [];

  for (let i = 0; i < NUM_CLASSES; i++) {
    const q = probs[i];              // uint8 0..255
    const percent = (q / 256) * 100; // scale 1/256

    classes.push({
      classId: i,
      label: labels[i],
      qValue: q,
      percent: percent
    });
  }

  // Ordenar do maior para o menor
  classes.sort((a, b) => b.qValue - a.qValue);

  console.log("\n===== TOP 20 CLASSES =====\n");

  for (let i = 0; i < 20; i++) {
    console.log(
      `Classe ${classes[i].classId} (${classes[i].label}) => ${classes[i].percent.toFixed(2)}% (q=${classes[i].qValue})`
    );
  }

  const sum = classes.reduce((acc, c) => acc + (c.qValue / 256), 0);
  console.log("Soma total =", sum);

})();
