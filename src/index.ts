import * as tf from '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { loadCSV } from './text_clear';

async function runPipeline() {
  try {

    // Data extraction
    const { textosLimpos, labels } = await loadCSV('liberal_conservatives.csv');


    // console.log(`Clean text data: ....`)
    // console.log(textosLimpos)
    console.log("Loading the USE model...");
    const encoder = await use.load();

    console.log("Starting batch embedding generation...");

    const TAMANHO_LOTE = 500; // Adjust this number depending on your RAM
    const tensoresTemporarios = []; // Array to store the chunks



    const embeddingsTensor = await encoder.embed(textosLimpos);
    const labelsTensor = tf.tensor2d(labels, [labels.length, 1]);
    // 3. Converting texts to Tensors
    const model = await configureNeuralNetwork();
    await trainingModel({ embeddingsTensor: embeddingsTensor as unknown as tf.Tensor2D, labelsTensor, model });
    console.log("Texts successfully converted to Tensors!");

    // Frees RAM memory (best practice in TensorFlow)
    embeddingsTensor.dispose();
    labelsTensor.dispose();
  } catch (error) {
    console.error("Error loading the model:", error);
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


  // 4. Neural network construction
  const model = tf.sequential();
  /***
   * [inputShape] is 512 because the USE model generates 512-dimensional embeddings
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
  await tf.ready(); // Ensures the Node.js backend is initialized
  await runPipeline();
})();
