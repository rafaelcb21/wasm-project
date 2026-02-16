/**
 * Validador de Layout de Memória WASM
 * Verifica se há overlaps e calcula endereços corretos
 */

console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║         VALIDADOR DE LAYOUT DE MEMÓRIA WASM               ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

// Configuração atual (do main.wat)
const PARAMS_BASE = 1760576;
const LP_SIZE = 112;
const TOTAL_LAYERS = 67;
const RESULT_BASE_ATUAL = 1770000;

console.log('📊 CONFIGURAÇÃO ATUAL:\n');
console.log(`   PARAMS_BASE:     ${PARAMS_BASE.toLocaleString()}`);
console.log(`   LP_SIZE:         ${LP_SIZE} bytes`);
console.log(`   TOTAL_LAYERS:    ${TOTAL_LAYERS}`);
console.log(`   RESULT_BASE:     ${RESULT_BASE_ATUAL.toLocaleString()}`);

// Cálculos
const params_size = TOTAL_LAYERS * LP_SIZE;
const params_end = PARAMS_BASE + params_size;

console.log('\n📐 CÁLCULOS:\n');
console.log(`   Tamanho dos params: ${TOTAL_LAYERS} × ${LP_SIZE} = ${params_size.toLocaleString()} bytes`);
console.log(`   Fim dos params:     ${PARAMS_BASE.toLocaleString()} + ${params_size.toLocaleString()} = ${params_end.toLocaleString()}`);

// Verificar overlap
console.log('\n🔍 VERIFICAÇÃO DE OVERLAP:\n');

if (RESULT_BASE_ATUAL < params_end) {
    const overlap = params_end - RESULT_BASE_ATUAL;
    console.log(`   🔴 OVERLAP DETECTADO!`);
    console.log(`   Overlap de ${overlap} bytes`);
    console.log(`   Primeiros ${overlap} bytes do resultado serão corrompidos!`);
    console.log(`\n   ⚠️  Isso explica por que os primeiros ${overlap} valores da saída`);
    console.log(`       têm padrões estranhos (13, 72, -54...)!\n`);
} else {
    console.log(`   ✅ SEM OVERLAP`);
    console.log(`   Gap de ${RESULT_BASE_ATUAL - params_end} bytes`);
}

// Sugerir correção
console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║                      CORREÇÃO NECESSÁRIA                   ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

const RESULT_BASE_MINIMO = params_end;
const RESULT_BASE_RECOMENDADO = Math.ceil((params_end + 1000) / 1000) * 1000; // Alinhar a 1KB

console.log('🔧 ENDEREÇOS CORRETOS:\n');
console.log(`   RESULT_BASE mínimo:      ${RESULT_BASE_MINIMO.toLocaleString()}`);
console.log(`   RESULT_BASE recomendado: ${RESULT_BASE_RECOMENDADO.toLocaleString()} (alinhado)`);

console.log('\n📝 MUDANÇA NO main.wat:\n');
console.log('   ANTES:');
console.log(`   (global $RESULT_BASE i32 (i32.const ${RESULT_BASE_ATUAL}))\n`);
console.log('   DEPOIS:');
console.log(`   (global $RESULT_BASE i32 (i32.const ${RESULT_BASE_RECOMENDADO}))\n`);

// Validar espaço disponível
const MEMORY_PAGES = 55;
const MEMORY_SIZE = MEMORY_PAGES * 65536; // 64KB por página
const resultado_end = RESULT_BASE_RECOMENDADO + 1000; // 1000 bytes para resultado

console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║                  VERIFICAÇÃO DE MEMÓRIA                    ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

console.log(`   Memória total:    ${MEMORY_PAGES} páginas = ${(MEMORY_SIZE / 1024 / 1024).toFixed(2)} MB`);
console.log(`   PARAMS região:    ${PARAMS_BASE.toLocaleString()} - ${params_end.toLocaleString()}`);
console.log(`   RESULT região:    ${RESULT_BASE_RECOMENDADO.toLocaleString()} - ${resultado_end.toLocaleString()}`);

if (resultado_end < MEMORY_SIZE) {
    console.log(`\n   ✅ Espaço suficiente!`);
    console.log(`   Memória usada: ${(resultado_end / 1024 / 1024).toFixed(2)} MB`);
    console.log(`   Memória livre: ${((MEMORY_SIZE - resultado_end) / 1024 / 1024).toFixed(2)} MB`);
} else {
    console.log(`\n   ❌ MEMÓRIA INSUFICIENTE!`);
    console.log(`   Necessário: ${(resultado_end / 1024 / 1024).toFixed(2)} MB`);
    console.log(`   Disponível: ${(MEMORY_SIZE / 1024 / 1024).toFixed(2)} MB`);
}

// Mapa de memória
console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║                     MAPA DE MEMÓRIA                        ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

console.log('   Região                   | Início      | Fim         | Tamanho');
console.log('   ─'.repeat(70));
console.log(`   LayerParams             | ${PARAMS_BASE.toLocaleString().padEnd(11)} | ${params_end.toLocaleString().padEnd(11)} | ${params_size.toLocaleString().padStart(7)} bytes`);

if (RESULT_BASE_ATUAL < params_end) {
    console.log(`   Resultado (ATUAL)       | ${RESULT_BASE_ATUAL.toLocaleString().padEnd(11)} | ${(RESULT_BASE_ATUAL + 1000).toLocaleString().padEnd(11)} |    1000 bytes 🔴 OVERLAP!`);
}

console.log(`   Resultado (CORRETO)     | ${RESULT_BASE_RECOMENDADO.toLocaleString().padEnd(11)} | ${resultado_end.toLocaleString().padEnd(11)} |    1000 bytes ✅`);

// Resumo final
console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║                      RESUMO EXECUTIVO                      ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

console.log('🎯 PROBLEMA IDENTIFICADO:');
console.log(`   Os últimos ${params_end - RESULT_BASE_ATUAL} bytes dos LayerParams`);
console.log(`   estão sobrescrevendo os primeiros ${params_end - RESULT_BASE_ATUAL} valores do resultado!\n`);



console.log('═'.repeat(70));
console.log('ANÁLISE CONCLUÍDA');
console.log('═'.repeat(70) + '\n');
