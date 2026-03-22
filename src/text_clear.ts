import fs from 'fs';
import csv from 'csv-parser';

// limpado os dados
function limparTextoReddit(text: string): string {
  return text
    // Remove URLs (http://... ou https://...)
    .replace(/https?:\/\/\S+/g, '')
    // Remove menções a usuários e subreddits (u/user, r/subreddit)
    .replace(/\b[ur]\/\w+/g, '')
    // Remove tags comuns de posts deletados
    .replace(/\[deleted\]|\[removed\]/gi, '')
    // Remove caracteres especiais pesados, mantendo pontuação básica
    .replace(/[^a-zA-Z0-9.,!? '"-]/g, ' ')
    // Remove espaços duplos ou quebras de linha excessivas
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function loadCSV(caminhoArquivo: string): Promise<{ textosLimpos: string[]; labels: number[] }> {
  return new Promise((resolve, reject) => {
    const textosLimpos: string[] = [];
    const labels: number[] = [];

    console.log("Iniciando a leitura e limpeza do dataset...");

    fs.createReadStream(caminhoArquivo)
      .pipe(csv()) // Transforma as linhas do CSV em objetos JavaScript
      .on('data', (linha) => {
        // console.log(`Linha ------->${JSON.stringify(linha)}`);
        // Supondo que as colunas no CSV do Kaggle se chamem 'post' e 'political_lean'
        // Você precisará checar o nome exato das colunas no arquivo baixado
        // Você acessa os dados usando linha['Nome Exato da Coluna']
        // No seu CSV lido, as colunas são: Title, Text, Political Lean, etc.
        const textoOriginal = linha.Title || linha.Text;
        const categoria = linha['Political Lean'];

        if (textoOriginal && categoria) {
          const textoLimpo = limparTextoReddit(textoOriginal);

          // Só adiciona se sobrou algum texto após a limpeza
          if (textoLimpo.length > 5) {
            textosLimpos.push(textoLimpo);

            // Converte a string 'Liberal'/'Conservative' para 0 e 1
            const labelNumerico = categoria.toLowerCase() === 'liberal' ? 0 : 1;
            labels.push(labelNumerico);
          }
        }
      })
      .on('end', () => {
        console.log(`Leitura concluída! Total de registros válidos: ${textosLimpos.length}`);
        console.log(`label_1: ${labels[0]}`);
        console.log(`textosLimpos_1: ${textosLimpos[0]}`);

        resolve({ textosLimpos, labels });
      })
      .on('error', (erro) => {
        console.error("Erro ao ler o CSV:", erro);
        reject(erro);
      });
  });
}

export { loadCSV };