package projektzpo.projekt;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class NeuralNetwork implements Serializable {
    private double bestAccuracy = 0.0;
    private double[][][] weights;
    private double[][] layerOutputs;
    private double[] errors;
    private int numberOfLayers;
    private int[] layers;
    private double[] targetOutput;
    private TfIdfVectorizer tfIdfVectorizer;
    private Map<String, Integer> categoryIndexMap;
    public void loadDataAndInitializeNetwork(Map<String, List<String[]>> trainingDataMap,
                                             int hiddenLayer1size,int hiddenLayer2size) {
        List<String> allCategory = new ArrayList<>();
        List<String> allDescription = new ArrayList<>();
        trainingDataMap.values().forEach(records -> {
            records.forEach(record -> {
                allCategory.add(record[0]);
                allDescription.add(record[1]);
            });

        });
        categoryIndexMap = new HashMap<>();
        int index = 0;
        for (String category : allCategory) {
            if (!categoryIndexMap.containsKey(category)) {
                categoryIndexMap.put(category, index++);
            }
        }
        int outputSize = categoryIndexMap.size();
        int inputSize = trainingDataMap.values()
                .stream()
                .flatMap(records -> records.stream())
                .mapToInt(record -> record[1].split("\\s+").length)
                .max()
                .orElse(0);
        addLayers(inputSize, outputSize, hiddenLayer1size, hiddenLayer2size);
        initializeRandomWeights();
        tfIdfVectorizer = initializeTfIdfVectorizer(allDescription);
    }

    public void trainNetwork(int numberOfEpochs, Map<String, List<String[]>> trainingDataMap,
                             double learningRate, int hiddenLayer1Size, int hiddenLayer2Size) {
        loadDataAndInitializeNetwork(trainingDataMap, hiddenLayer1Size, hiddenLayer2Size);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("wyniki.txt"))) {
            for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
                for (List<String[]> records : trainingDataMap.values()) {
                    records.forEach(record -> backpropagate(record[0], record[1], learningRate));
                }

                double accuracy = testNetwork(trainingDataMap);
                System.out.println("Epoch " + epoch + " - Accuracy: " + accuracy);
                writer.write("Epoch " + epoch + " - Accuracy: " + accuracy + "\n");
                // Dodanie wyników predykcji do pliku
                for (Map.Entry<String, List<String[]>> entry : trainingDataMap.entrySet()) {
                    List<String[]> records = entry.getValue();
                    for (String[] record : records) {
                        String trueCategory = record[0];
                        String description = record[1];
                        forwardPropagation(description);
                        int predictedCategoryIndex = getPredictedCategoryIndex();
                        String predictedCategory = getCategoryFromIndex(predictedCategoryIndex);
                        String trueCategoryWord = null;
                        Integer trueCategoryIndex = categoryIndexMap.get(trueCategory);
                        if (trueCategoryIndex != null) {
                            trueCategoryWord = getCategoryFromIndex(trueCategoryIndex);
                        }
                        writer.write("True: " + trueCategoryWord + " Prediction: " + predictedCategory + "\n");
                    }
                }

                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void initializeRandomWeights() {
        Random random = new Random();
        for (int layer = 0; layer < numberOfLayers - 1; layer++) {
            for (int neuronIndex = 0; neuronIndex < layers[layer]; neuronIndex++) {
                for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < layers[layer + 1]; nextLayerNeuronIndex++) {
                    double weight = random.nextGaussian() * Math.sqrt(1.0 / (layers[layer] + layers[layer + 1]));
                    weights[layer][neuronIndex][nextLayerNeuronIndex] = weight;
                }
            }
        }
    }

    private void backpropagate(String category, String description,double learingRate) {
        forwardPropagation(description);
        calculateLoss(category);
        backwardPropagation(learingRate);
    }

    private void backwardPropagation(double learningRate) {

        for (int layer = numberOfLayers - 1; layer > 0; layer--) {
            for (int neuronIndex = 0; neuronIndex < layers[layer]; neuronIndex++) {
                updateWeights(layer, neuronIndex, learningRate);
            }
        }
    }

    private void updateWeights(int layer, int neuronIndex, double learningRate) {
        if (neuronIndex < errors.length) {
            double gradient = errors[neuronIndex] * activationFunctionDerivative(layerOutputs[layer][neuronIndex]);
            for (int prevNeuronIndex = 0; prevNeuronIndex < layers[layer - 1]; prevNeuronIndex++) {
                updateWeight(layer, prevNeuronIndex, neuronIndex, gradient, learningRate);
            }
        }
    }

    private void updateWeight(int layer, int prevNeuronIndex, int neuronIndex, double gradient, double learningRate) {
        double weightUpdate = learningRate * gradient * layerOutputs[layer - 1][prevNeuronIndex];
        if (neuronIndex < weights[layer - 1][prevNeuronIndex].length) {
            weights[layer - 1][prevNeuronIndex][neuronIndex] -= weightUpdate;
        }
    }

    private TfIdfVectorizer initializeTfIdfVectorizer(List<String> allDescriptions) {
        TfIdfVectorizer tfIdfVectorizer = new TfIdfVectorizer();
        tfIdfVectorizer.fit(allDescriptions);
        return tfIdfVectorizer;
    }

    public class TfIdfVectorizer {
        private Map<String, Integer> documentFrequencyMap;
        private List<Map<String, Double>> tfIdfVectors;
        public void fit(List<String> documents) {
            documentFrequencyMap = new HashMap<>();
            tfIdfVectors = new ArrayList<>();

            for (String document : documents) {
                Map<String, Double> tfIdfVector = new HashMap<>();
                String[] words = document.replaceAll("[^a-zA-Z\\s]", "").toLowerCase().split("\\s+");

                Map<String, Integer> termFrequencyMap = new HashMap<>();
                for (String word : words) {
                    termFrequencyMap.put(word, termFrequencyMap.getOrDefault(word, 0) + 1);
                }
                for (String term : termFrequencyMap.keySet()) {
                    documentFrequencyMap.put(term, documentFrequencyMap.getOrDefault(term, 0) + 1);
                }
                for (String term : termFrequencyMap.keySet()) {
                    double tf = (double) termFrequencyMap.get(term) / words.length;
                    double idf = Math.log((double) documents.size() / (documentFrequencyMap.get(term) + 1));
                    tfIdfVector.put(term, tf * idf);
                }
                tfIdfVectors.add(tfIdfVector);
            }
        }

        public double[] transform(String document) {
            Map<String, Double> tfIdfVector = new HashMap<>();
            String[] words = document.split("\\s+");
            Map<String, Integer> termFrequencyMap = new HashMap<>();
            for (String word : words) {
                termFrequencyMap.put(word, termFrequencyMap.getOrDefault(word, 0) + 1);
            }
            for (String term : termFrequencyMap.keySet()) {
                double tf = (double) termFrequencyMap.get(term) / words.length;
                double idf = Math.log((double) tfIdfVectors.size() / (documentFrequencyMap.getOrDefault(term, 0) + 1));
                tfIdfVector.put(term, tf * idf);
            }
            double[] result = new double[tfIdfVectors.size()];
            for (int i = 0; i < tfIdfVectors.size(); i++) {
                result[i] = tfIdfVector.getOrDefault(String.valueOf(i), 0.0);
            }
            return result;
        }
    }
    private void initializeInputLayer(List<String> inputWords, TfIdfVectorizer tfIdfVectorizer) {
        double[] tfIdfVector = tfIdfVectorizer.transform(String.join(" ", inputWords));
        if (layerOutputs[0].length < tfIdfVector.length) {
            layerOutputs[0] = new double[tfIdfVector.length];
        }
        System.arraycopy(tfIdfVector, 0, layerOutputs[0], 0, tfIdfVector.length);
    }
    private void forwardPropagation(String description) {
        initializeInputLayer(Arrays.asList(description.split("\\s+")), tfIdfVectorizer);
        for (int layer = 1; layer < numberOfLayers; layer++) {
            for (int neuronIndex = 0; neuronIndex < layers[layer]; neuronIndex++) {
                double sum = 0.0;
                for (int prevNeuronIndex = 0; prevNeuronIndex < layers[layer - 1]; prevNeuronIndex++) {
                    sum += weights[layer - 1][prevNeuronIndex][neuronIndex] * layerOutputs[layer - 1][prevNeuronIndex];
                }
                layerOutputs[layer][neuronIndex] = activationFunction(sum);
            }
        }
    }
    private void calculateLoss(String batchCategory) {
        for (int i = 0; i < layers[numberOfLayers - 1]; i++) {
            Integer categoryIndex = categoryIndexMap.get(batchCategory);
            if (categoryIndex != null) {
                int targetIndex = categoryIndex.intValue();
                errors[i] = layerOutputs[numberOfLayers - 1][i] - targetOutput[targetIndex];
            }
        }
    }

    public double testNetwork(Map<String, List<String[]>> testDataMap) {
        int correctPredictions = 0;
        int totalPredictions = 0;
        for (Map.Entry<String, List<String[]>> entry : testDataMap.entrySet()) {
            List<String[]> records = entry.getValue();
            for (String[] record : records) {
                String trueCategory = record[0];
                String description = record[1];
                forwardPropagation(description);
                int predictedCategoryIndex = getPredictedCategoryIndex();
                String predictedCategory = getCategoryFromIndex(predictedCategoryIndex);
                String trueCategoryWord = null;
                Integer trueCategoryIndex = categoryIndexMap.get(trueCategory);
                if (trueCategoryIndex != null) {
                    trueCategoryWord = getCategoryFromIndex(trueCategoryIndex);
                } else {
                    //System.out.println("Nie znaleziono kategorii: " + trueCategory);
                }
                //System.out.println("True: " + categoryIndexMap.get(trueCategory) + " Prediction: " + predictedCategoryIndex);
                //System.out.println("True: " + trueCategoryWord + " Prediction: " + predictedCategory);

                if (trueCategoryIndex != null && trueCategoryIndex == predictedCategoryIndex) {
                    correctPredictions++;
                }
                totalPredictions++;
            }
        }
        return (double) correctPredictions / totalPredictions * 100.0;
    }

    private String getCategoryFromIndex(int index) {
        for (Map.Entry<String, Integer> entry : categoryIndexMap.entrySet()) {
            if (entry.getValue() == index) {
                return entry.getKey();
            }
        }
        return "Unknown";
    }

    private int getPredictedCategoryIndex() {
        int outputLayerIndex = numberOfLayers - 1;
        double[] outputLayer = layerOutputs[outputLayerIndex];
        double maxProbability = outputLayer[0];  // Inicjalizuj przed pętlą
        int predictedCategoryIndex = 0;
        for (int i = 1; i < outputLayer.length; i++) {
            if (outputLayer[i] > maxProbability) {
                maxProbability = outputLayer[i];
                predictedCategoryIndex = i;
            }
        }
        return predictedCategoryIndex;
    }

    private void addLayers(int inputSize, int outputSize, int hiddenLayer1Size, int hiddenLayer2Size) {
        numberOfLayers = 4;
        layers = new int[]{inputSize, hiddenLayer1Size, hiddenLayer2Size, outputSize};
        weights = new double[numberOfLayers - 1][][];
        layerOutputs = new double[numberOfLayers][];
        for (int layer = 0; layer < numberOfLayers - 1; layer++) {
            weights[layer] = new double[layers[layer]][layers[layer + 1]];
        }
        for (int layer = 0; layer < numberOfLayers; layer++) {
            layerOutputs[layer] = new double[layers[layer]];
        }
        errors = new double[outputSize];
        targetOutput = new double[outputSize];
    }

    private double activationFunction(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double activationFunctionDerivative(double x) {
        return x * (1.0 - x);
    }
}