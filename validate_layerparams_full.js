const fs = require('fs');

(async () => {
    console.log('🔍 EXPORTANDO TODOS OS LAYERPARAMS PARA JSON\n');

    const wasmBuffer = fs.readFileSync("main.wasm");
    const wasmModule = await WebAssembly.instantiate(wasmBuffer);
    const instance = wasmModule.instance;
    const memory = instance.exports.memory;

    const PARAMS_BASE = 1760576;
    const LP_SIZE = 112;
    const NUM_LAYERS = 67;

    const FIELD = {
        op_type: 0,
        act: 1,
        flags: 2,
        in_ptr: 3,
        out_ptr: 4,
        in_h: 5,
        in_w: 6,
        cin: 7,
        cout: 8,
        kh: 9,
        kw: 10,
        stride_h: 11,
        stride_w: 12,
        dil_h: 13,
        dil_w: 14,
        pad_t: 15,
        pad_b: 16,
        pad_l: 17,
        pad_r: 18,
        wptr: 19,
        bias_ptr: 20,
        mul_ptr: 21,
        q6_ptr: 22,
        zx: 23,
        zw: 24,
        zy: 25,
        out_h: 26,
        out_w: 27
    };


    const OP_NAMES = {
        1: 'CONV',
        2: 'DEPTHWISE',
        3: 'FC',
        4: 'ADD',
        5: 'MEAN',
        6: 'SOFTMAX',
        7: 'QUANTIZE'
    };

    const layers = [];

    for (let i = 0; i < NUM_LAYERS; i++) {

        const base = PARAMS_BASE + i * LP_SIZE;
        const p = new Int32Array(memory.buffer, base, 28);

        const op = p[FIELD.op_type];
        const typeName = OP_NAMES[op] || `UNKNOWN(${op})`;

        const layerInfo = {
            index: i,
            op_type: op,
            type: typeName,
            raw: Array.from(p),
            fields: {}
        };

        // salvar todos os campos nomeados
        for (const [name, idx] of Object.entries(FIELD)) {
            layerInfo.fields[name] = p[idx];
        }

        layers.push(layerInfo);
    }

    const report = {
        timestamp: new Date().toISOString(),
        params_base: PARAMS_BASE,
        layer_size_bytes: LP_SIZE,
        total_layers: NUM_LAYERS,
        layers
    };

    fs.writeFileSync("layerparam_validation_full.json", JSON.stringify(report, null, 2));

    console.log('✅ layerparam_validation_full.json gerado com sucesso\n');
    console.log('📌 Verifique especialmente a layer 66 para entender o op_type real.\n');

})();
