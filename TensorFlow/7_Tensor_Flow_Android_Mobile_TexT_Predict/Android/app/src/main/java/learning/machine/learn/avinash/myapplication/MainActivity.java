package learning.machine.learn.avinash.myapplication;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
public class MainActivity extends AppCompatActivity {

    EditText editText;
    TextView textView;

    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/optimized_text_predictor.pb";
    private static final String INPUT_NODE = "x_input";
    private static final long[] INPUT_SHAPE = {1, 3, 1};
    private static final String OUTPUT_NODE = "y_output";
    private TensorFlowInferenceInterface inferenceInterface;

    private String[] vocabDict = {};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        editText = findViewById(R.id.edit_text);
        textView = findViewById(R.id.text_view);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        try {
            String fileString = readFile("vocab_dict.txt");
            vocabDict = fileString.split(" ");
            Log.d("MainActivity", vocabDict[10]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void predictTextAction(View view) {
        String text = editText.getText().toString();
        Log.d("Predict  Test" , text);
        if (!text.isEmpty()) {
            formatInput(text);
        }
    }

    private void formatInput(String input) {
        String[] splitString = input.split(" ");
        if (splitString.length < 4) {
            return;
        }

        String firstString = splitString[splitString.length - 3];
        String secondString = splitString[splitString.length - 2];
        String thirdString = splitString[splitString.length - 1];

        float first = getIndex(firstString);
        float second = getIndex(secondString);
        float third = getIndex(thirdString);

        float[] modelInput = {first, second, third};
        runInference(modelInput);
    }

    private float getIndex(String input) {
        for (int i = 0; i < vocabDict.length; i++) {
            if (vocabDict[i].equals(input)) {
                return (float) i;
            }
        }
        return (float) 0;
    }

    private void runInference(float[] input) {
        inferenceInterface.feed(INPUT_NODE, input, INPUT_SHAPE);
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        float[] results = new float[vocabDict.length];
        inferenceInterface.fetch(OUTPUT_NODE, results);

        interpretResults(results);
    }

    private void interpretResults(float[] results) {
        float max = results[0];
        int maxIndex = 0;
        for (int i = 0; i < results.length; i++) {
            if (results[i] > max) {
                max = results[i];
                maxIndex = i;
            }
        }
        textView.setText(vocabDict[maxIndex]);
    }

    private String readFile(String filename) throws IOException {
        InputStream inputStream = getAssets().open(filename);
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        try {
            StringBuilder stringBuilder = new StringBuilder();
            String line = bufferedReader.readLine();
            while (line != null) {
                stringBuilder.append(line);
                stringBuilder.append("\n");
                line = bufferedReader.readLine();
            }
            return stringBuilder.toString();
        } finally {
            bufferedReader.close();
        }
    }
}
