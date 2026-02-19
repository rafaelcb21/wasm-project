import { memory_simulated } from "./memory_simulated.js";

const view = new DataView(memory_simulated.buffer);

/**
 * Lê 1 byte
 */
export function readByte(addr) {
    if (addr < 0 || addr >= memory_simulated.length) {
        throw new Error("Endereço fora da memória");
    }

    return view.getUint8(addr);
}

/**
 * Lê 1 byte como Int8 (signed)
 */
export function readInt8(addr) {
    if (addr < 0 || addr >= memory_simulated.length) {
        throw new Error("Endereço fora da memória");
    }

    return view.getInt8(addr);
}

/**
 * Lê 4 bytes little-endian como Uint32
 */
export function readInt32(addr) {
    if (addr < 0 || addr + 3 >= memory_simulated.length) {
        throw new Error("Endereço fora da memória");
    }

    return view.getUint32(addr, true); // little-endian
}

/**
 * Lê 4 bytes little-endian como Int32 (signed)
 */
export function readInt32Signed(addr) {
    if (addr < 0 || addr + 3 >= memory_simulated.length) {
        throw new Error("Endereço fora da memória");
    }

    return view.getInt32(addr, true);
}

/**
 * Retorna os 4 bytes crus
 */
export function readRaw4(addr) {
    if (addr < 0 || addr + 3 >= memory_simulated.length) {
        throw new Error("Endereço fora da memória");
    }

    return [
        view.getUint8(addr),
        view.getUint8(addr + 1),
        view.getUint8(addr + 2),
        view.getUint8(addr + 3),
    ];
}

/**
 * Debug de endereço
 */
export function debugAddress(addr) {
    console.log("Addr:", addr);
    console.log("Byte:", readByte(addr));

    if (addr + 3 < memory_simulated.length) {
        console.log("Int32 unsigned:", readInt32(addr));
        console.log("Int32 signed  :", readInt32Signed(addr));
    }
}
