package projektzpo.projekt;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import java.io.IOException;

public class NeuralNetworkApp extends Application {

    //Start aplikacji JavaFX
    public static void main(String[] args) {
        launch(args);
    }

    @Override// Wywoływane przy uruchamianiu aplikacji
    public void start(Stage primaryStage) throws IOException {
        //Wczytanie pliku FXML
        FXMLLoader loader = new FXMLLoader(getClass().getResource("neural-network-view.fxml"));
        Parent root = loader.load();
        //Utworzenie nowego okna
        primaryStage.setTitle("Neural Network App");
        primaryStage.setScene(new Scene(root, 300, 200));
        primaryStage.show();
    }
}