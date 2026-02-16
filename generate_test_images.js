const fs = require('fs');

/**
 * Gerador de Imagens de Teste
 * Cria imagens sintéticas para testar a rede
 */

function createTestImage(type) {
    const width = 224;
    const height = 224;
    const channels = 3;
    const buffer = Buffer.alloc(width * height * channels);
    
    for (let i = 0; i < width * height; i++) {
        let r, g, b;
        
        switch(type) {
            case 'black':
                r = g = b = 0;
                break;
                
            case 'white':
                r = g = b = 255;
                break;
                
            case 'red':
                r = 255; g = 0; b = 0;
                break;
                
            case 'green':
                r = 0; g = 255; b = 0;
                break;
                
            case 'blue':
                r = 0; g = 0; b = 255;
                break;
                
            case 'gray':
                r = g = b = 128;
                break;
                
            case 'gradient':
                const row = Math.floor(i / width);
                const value = Math.floor((row / height) * 255);
                r = g = b = value;
                break;
                
            case 'checkerboard':
                const cx = (i % width) >> 4;  // div por 16
                const cy = Math.floor(i / width) >> 4;
                const isWhite = (cx + cy) % 2 === 0;
                r = g = b = isWhite ? 255 : 0;
                break;
                
            case 'noise':
                r = Math.floor(Math.random() * 256);
                g = Math.floor(Math.random() * 256);
                b = Math.floor(Math.random() * 256);
                break;
                
            default:
                r = g = b = 128;
        }
        
        buffer[i * 3 + 0] = r;
        buffer[i * 3 + 1] = g;
        buffer[i * 3 + 2] = b;
    }
    
    return buffer;
}

console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║           GERADOR DE IMAGENS DE TESTE 224×224×3            ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

const testImages = [
    { name: 'test_black.raw', type: 'black', desc: 'Imagem toda preta (0,0,0)' },
    { name: 'test_white.raw', type: 'white', desc: 'Imagem toda branca (255,255,255)' },
    { name: 'test_red.raw', type: 'red', desc: 'Imagem toda vermelha (255,0,0)' },
    { name: 'test_green.raw', type: 'green', desc: 'Imagem toda verde (0,255,0)' },
    { name: 'test_blue.raw', type: 'blue', desc: 'Imagem toda azul (0,0,255)' },
    { name: 'test_gray.raw', type: 'gray', desc: 'Imagem cinza médio (128,128,128)' },
    { name: 'test_gradient.raw', type: 'gradient', desc: 'Gradiente vertical (0→255)' },
    { name: 'test_checkerboard.raw', type: 'checkerboard', desc: 'Tabuleiro de xadrez 16×16' },
    { name: 'test_noise.raw', type: 'noise', desc: 'Ruído aleatório RGB' }
];

console.log('Gerando imagens de teste...\n');

testImages.forEach(img => {
    const buffer = createTestImage(img.type);
    fs.writeFileSync(img.name, buffer);
    console.log(`✅ ${img.name.padEnd(25)} - ${img.desc}`);
    console.log(`   Tamanho: ${buffer.length} bytes (${224}×${224}×3)\n`);
});

console.log('═'.repeat(64));
console.log('✨ Todas as imagens de teste foram geradas!\n');

console.log('📋 Como usar:');
console.log('   1. Renomeie uma imagem de teste:');
console.log('      cp test_black.raw aviao_uint8.raw');
console.log('   2. Execute o teste:');
console.log('      node test_mobilenetv2.js');
console.log('   3. Compare os resultados\n');

console.log('🎯 O que esperar:');
console.log('   • Imagens diferentes devem produzir saídas diferentes');
console.log('   • Se todas produzem a mesma saída → problema na rede');
console.log('   • Se black/white produzem resultados razoáveis → problema nos pesos\n');
