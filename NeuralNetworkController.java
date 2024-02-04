package projektzpo.projekt;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class NeuralNetworkController {
    @FXML
    private TextField numberOfEpochsField;
    @FXML
    private TextField learningRatesField;
    @FXML
    private TextField hiddenLayer1SizesField;
    @FXML
    private TextField hiddenLayer2SizesField;
    @FXML
    private Label accuracyLabel;
    @FXML
    void initialize() {
        neuralNetwork = new NeuralNetwork();
    }
    private NeuralNetwork neuralNetwork;
    private Map<String, List<String[]>> trainingDataMap;
    private Map<String, List<String[]>> testingDataMap;
    private void showAlert(String title, String content) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }

    @FXML
    void trainNetwork(ActionEvent actionEvent) {
        try {
            if (numberOfEpochsField.getText().isEmpty() || learningRatesField.getText().isEmpty() ||
                    hiddenLayer1SizesField.getText().isEmpty() || hiddenLayer2SizesField.getText().isEmpty()) {
                showAlert("Missing Values", "Please enter all the required values.");
                return;
            }
            int numberOfEpochs = Integer.parseInt(numberOfEpochsField.getText());
            double learningRate = Double.parseDouble(learningRatesField.getText());
            int hiddenLayer1Size = Integer.parseInt(hiddenLayer1SizesField.getText());
            int hiddenLayer2Size = Integer.parseInt(hiddenLayer2SizesField.getText());
            neuralNetwork.trainNetwork(numberOfEpochs, trainingDataMap, learningRate, hiddenLayer1Size, hiddenLayer2Size);
            showAlert("Training Complete", "Neural network training completed successfully.");
        } catch (NumberFormatException e) {
            showAlert("Invalid input", "Please enter valid numerical values.");
        }
    }
    @FXML
    void testNetwork(ActionEvent actionEvent) {
        try {
          //  System.out.println("Testing Data Map:");
            for (Map.Entry<String, List<String[]>> entry : testingDataMap.entrySet()) {
                String category = entry.getKey();
                List<String[]> records = entry.getValue();
             //   System.out.println("Category: " + category);
                for (String[] record : records) {
                 //   System.out.println("  " + Arrays.toString(record));
                }
            }
            double accuracy = neuralNetwork.testNetwork(testingDataMap);
            showAlert("Testing Complete", "Neural network testing completed successfully.");
            accuracyLabel.setText("New Accuracy: " + accuracy + "%");

        } catch (NumberFormatException e) {
            showAlert("Invalid input", "Please check error.");
        }
    }

    @FXML
    void loadTrainingData(ActionEvent actionEvent) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Training Data File");
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("CSV Files", "*.csv")
        );
        // Pokaż okno dialogowe i pobierz wybrany plik
        File selectedFile = fileChooser.showOpenDialog(new Stage());
        if (selectedFile != null) {
            try {
                // Wczytaj dane z pliku CSV
                List<String[]> data = readCSVFile(selectedFile);
                // Mapa do przechowywania danych treningowych
                trainingDataMap = prepareTrainingData(data);
                // Wyświetl informacje w konsoli
            } catch (IOException e) {
                showAlert("Error", "An error occurred while loading the training data.");
            }
        }
    }

    @FXML
    void loadTestData(ActionEvent actionEvent) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Test Data File");
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("CSV Files", "*.csv")
        );
        // Pokaż okno dialogowe i pobierz wybrany plik
        File selectedFile = fileChooser.showOpenDialog(new Stage());
        if (selectedFile != null) {
            try {
                // Wczytaj dane z pliku CSV
                List<String[]> data = readCSVFile(selectedFile);
                // Mapa do przechowywania danych testowych
                testingDataMap = prepareTestingData(data);
            } catch (IOException e) {
                showAlert("Error", "An error occurred while loading the test data.");
            }
        }
    }

    private Map<String, List<String[]>> prepareTrainingData(List<String[]> data) {
        Map<String, List<String[]>> preparedData = new HashMap<>();
        for (String[] record : data) {
            String category = getCategoryFromRecord(record);
            preparedData.computeIfAbsent(category, k -> new ArrayList<>()).add(record);
        }
        return preparedData;
    }

    private Map<String, List<String[]>> prepareTestingData(List<String[]> data) {
        Map<String, List<String[]>> preparedData = new HashMap<>();
        for (String[] record : data) {
            String category = getCategoryFromRecord(record);
            preparedData.computeIfAbsent(category, k -> new ArrayList<>()).add(record);
        }
        return preparedData;
    }

    private String getCategoryFromRecord(String[] record) {
        List<String> categoryKeywords = Arrays.asList("Household", "Books", "Clothing & Accessories", "Electronics");

        for (String keyword : categoryKeywords) {
            if (record[1].contains(keyword)) {
                return keyword;
            }
        }
        return "Other";
    }

    // Metoda do czytania danych z pliku CSV
    private List<String[]> readCSVFile(File file) throws IOException {
        List<String[]> data = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            boolean firstLine = true; // Dodane pole do oznaczenia pierwszej linii
            while ((line = reader.readLine()) != null) {
                if (firstLine) {
                    // Pomijaj pierwszą linię (zakładamy, że zawiera nagłówki kolumn)
                    firstLine = false;
                    continue;
                }
                String[] values = line.split(",");
                data.add(values);
            }
        }
        return data;
    }
}
