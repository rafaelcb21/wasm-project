import { readByte, readInt32, debugAddress, readInt32Signed, readRaw4 } from "./memory_reader.js";

//console.log(readByte(2049));       // peso int8
//console.log(readInt32(1664096));   // bias int32
console.log(readInt32Signed(1664100));   // bias int32
console.log(readRaw4(1664100));   // -30175 => bias int32

//debugAddress(2049);
