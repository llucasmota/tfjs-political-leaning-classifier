# tfjs-political-leaning-classifier

[![TypeScript](https://img.shields.io/badge/TypeScript-5.9.3-blue.svg)](https://www.typescriptlang.org/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Node_Backend-orange)](https://js.tensorflow.org/)

A Machine Learning model (Natural Language Processing) designed to analyze and predict the political leaning (Liberal vs. Conservative) of texts based on semantic context.

This project builds, configures, and trains a **Dense Neural Network (Supervised)** exclusively in TypeScript and Node.js.

## 🎯 Architectural Goal
The main technological goal of this project is to demonstrate how to build text classification pipelines from scratch using the JavaScript ecosystem. The system is able to "read" Reddit titles containing sarcasm and political opinions, and accurately infer the author's leaning through semantic vector calculation.

## 🛠 Tech Stack
The stack was chosen to avoid traditional Python-based libraries, paving the way for scalable, event-driven AI applications built entirely on Node.js:

*   **TypeScript** - For type safety and maintainability.
*   **TensorFlow.js (`@tensorflow/tfjs-node`)** - The core engine of the Neural Network, leveraging C++ bindings under the hood for CPU/GPU acceleration.
*   **Universal Sentence Encoder (`@tensorflow-models/universal-sentence-encoder`)** - A robust NLP model that encodes any text into a semantic vector with *512 dimensions*.
*   **Node.js & csv-parser** - For parsing and extracting raw tabular data.

## 📊 Data Source
The training data comes from real user interactions on **Reddit**, extracted via Kaggle.
*   **Official Dataset Link:** [Liberals vs Conservatives on Reddit (13,000 Posts)](https://www.kaggle.com/datasets/neelgajare/liberals-vs-conservatives-on-reddit-13000-posts/data)
*   **Features Used:** The model correlates the `Title` column against the supervised label column (`Political Lean`).

## 🚀 How to Run Locally

### Prerequisites
*   Node.js (version 20+ recommended)
*   The CSV dataset placed in the project root (`liberal_conservatives.csv`)

### Quick Start
Install all dependencies:
```bash
npm install
```

Start the full pipeline (Extraction -> Processing -> Training -> Evaluation):
```bash
npm run dev
```

## 🔮 Future Roadmap (Next Level Specs)
This system is planned to transcend a simple Node script and evolve into a fully-fledged MLOps pipeline. Next steps include:

1.  **Scalable Vector Database:** Integrating *Pinecone* or *PostgreSQL (pgvector)* to store the 512-dimension Embeddings computed ahead of time. This removes bottlenecks and prevents recalculating the *Universal Sentence Encoder* on every training cycle.
2.  **Real-Time Inference API:** Creating an Express/Fastify API endpoint where users can send random text for the system to predict their political leaning percentage synchronously, using a pre-loaded trained model in-memory.
3.  **Advanced NLP Processing:** Implementing a robust cleaning pipeline using Regex and Stop-Words removal (eliminating links, "Edit:" tags, and URLs) to ensure the signal sent to the embeddings is as pure as possible.
4.  **Feature Engineering:** Expanding the Sequential Neural Network into a Multi-Input model, crossing text intelligence (from the *Title*) with tabular numeric data like the Reddit *Score* and *Upvotes* metrics.
