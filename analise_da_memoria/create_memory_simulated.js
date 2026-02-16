import fs from "fs";

async function generateMemorySimulation(wasmFile, outputFile) {
    const buffer = fs.readFileSync(wasmFile);

    // NÃO criar memória manual
    const wasm = await WebAssembly.instantiate(buffer, {
        env: { log: () => {} }
    });

    const exports = wasm.instance.exports;

    // 🔥 pegar memória real do módulo
    const memory = exports.memory;

    const memBuffer = memory.buffer;
    const mem8 = new Uint8Array(memBuffer);
    const totalBytes = mem8.length;

    const simulated = new Uint8Array(totalBytes);

    const slot0 = exports.get_slot0_base();

    for (let i = 0; i < totalBytes; i++) {

        // FREE inicial
        if (i < exports.get_weights_base()) {
            simulated[i] = 0;
            continue;
        }

        // SLOTs e arena → zera
        if (i >= slot0) {
            simulated[i] = 0;
            continue;
        }

        // copia byte real do wasm
        simulated[i] = mem8[i];
    }

    const fileContent =
        "export const memory_simulated = new Uint8Array(" +
        JSON.stringify(Array.from(simulated)) +
        ");\n";

    fs.writeFileSync(outputFile, fileContent);

    console.log("✅ memory_simulated.js gerado com sucesso");
}

generateMemorySimulation("main.wasm", "memory_simulated.js");
