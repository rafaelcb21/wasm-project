import fs from "fs";

async function analyzeWasmRuntime(filename) {
    const buffer = fs.readFileSync(filename);

    const memory = new WebAssembly.Memory({ initial: 56 });

    const imports = {
        env: {
            memory,
            log: () => {}
        }
    };

    const wasm = await WebAssembly.instantiate(buffer, imports);
    const exports = wasm.instance.exports;

    const totalBytes = memory.buffer.byteLength;

    const bases = [
        { name: "WEIGHTS_BASE", value: exports.get_weights_base(), elemSize: 1 },
        { name: "BIAS_BASE", value: exports.get_bias_base(), elemSize: 4 },
        { name: "MUL_BASE", value: exports.get_mul_base(), elemSize: 4 },
        { name: "SHIFT_BASE", value: exports.get_shift_base(), elemSize: 4 },
        { name: "Q6_BASE", value: exports.get_q6_base(), elemSize: 4 },
        { name: "PARAMS_BASE", value: exports.get_params_base(), elemSize: 4 },
        { name: "SLOT0_BASE", value: exports.get_slot0_base(), elemSize: 1 },
        { name: "SLOT1_BASE", value: exports.get_slot1_base(), elemSize: 1 },
        { name: "SLOT2_BASE", value: exports.get_slot2_base(), elemSize: 1 },
    ];

    bases.sort((a, b) => a.value - b.value);

    console.log("\n===== WASM MEMORY MAP (WITH ELEMENT SIZE) =====\n");
    console.log("Total bytes:", totalBytes);
    console.log("");

    function printRange(name, start, end, elemSize) {
        const size = end - start + 1;
        const elements = Math.floor(size / elemSize);

        console.log(
            `${name.padEnd(22)}: ${start} → ${end} ` +
            `(${size} bytes) (${elemSize}B/elem, ~${elements} elems)`
        );
    }

    // FREE inicial
    if (bases[0].value > 0) {
        printRange("FREE", 0, bases[0].value - 1, 1);
    }

    for (let i = 0; i < bases.length; i++) {
        const start = bases[i].value;
        const end =
            i < bases.length - 1
                ? bases[i + 1].value - 1
                : totalBytes - 1;

        printRange(
            bases[i].name,
            start,
            end,
            bases[i].elemSize
        );
    }

    console.log("\n===============================================\n");
}

analyzeWasmRuntime("main.wasm");
