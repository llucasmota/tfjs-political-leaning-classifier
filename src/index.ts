import * as tf from '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { loadCSV } from './text_clear';

async function runPipeline() {
  try {

    // Extração dos dados
    const { textosLimpos, labels } = await loadCSV('liberal_conservatives.csv');


    // console.log(`Dados de textos limpos: ....`)
    // console.log(textosLimpos)
    console.log("Carregando o modelo USE...");
    const encoder = await use.load();

    console.log("Iniciando a geração de embeddings em lotes...");

    const TAMANHO_LOTE = 500; // Ajuste este número dependendo da sua memória RAM
    const tensoresTemporarios = []; // Array para guardar os pedaços



    const embeddingsTensor = await encoder.embed(textosLimpos);
    const labelsTensor = tf.tensor2d(labels, [labels.length, 1]);
    // 3. Convertendo textos para Tensores
    const model = await configureNeuralNetwork();
    await trainingModel({ embeddingsTensor: embeddingsTensor as unknown as tf.Tensor2D, labelsTensor, model });
    console.log("Textos convertidos para Tensores com sucesso!");

    // Libera a memória da RAM (boa prática no TensorFlow)
    embeddingsTensor.dispose();
    labelsTensor.dispose();
  } catch (error) {
    console.error("Erro ao carregar o modelo:", error);
  }
}

// // Example: Create a simple tensor and print it
// const shape = [2, 3]; // 2 rows, 3 columns
// const data = [1, 2, 3, 4, 5, 6];
// const tensor = tf.tensor(data, shape);

// tensor.print(); // Expected output: [[1, 2, 3], [4, 5, 6]]

// console.log('TensorFlow.js is set up with TypeScript in Node.js!');

async function trainingModel({ embeddingsTensor, labelsTensor, model }: { embeddingsTensor: tf.Tensor2D, labelsTensor: tf.Tensor2D, model: tf.Sequential }) {
  model.fit(embeddingsTensor, labelsTensor, {
    epochs: 15,
    batchSize: 64,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: Loss = ${logs?.loss}, Accuracy = ${logs?.acc}`);

      }
    }
  });

}

async function configureNeuralNetwork(): Promise<tf.Sequential> {


  // 4. Construção da rede neural
  const model = tf.sequential();
  /***
   * é 512 o [inputShape] porque o modelo USE gera embeddings de 512 dimensões
   * 
   */
  model.add(tf.layers.dense({ inputShape: [512], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
  }))

  model.add(tf.layers.dense({
    units: 32,
    activation: 'relu',
  }))



  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}



// Asynchronous self-executing function
(async () => {
  await tf.ready(); // Garante que o backend do Node.js está inicializado
  await runPipeline();
})();
