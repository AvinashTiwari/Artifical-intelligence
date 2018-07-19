package learning.machine.learn.avinash.weather_predection;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import android.view.View;
import android.widget.CheckBox;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    CheckBox maxTempCheckBox, minTempCheckBox, meanTempCheckBox, precipCheckBox;
    TextView resultsTextView;

    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/optimized_weather_prediction.pb";
    private static final String INPUT_NODE = "x_input";
    private static final int[] INPUT_SHAPE = {1, 4};
    private static final String OUTPUT_NODE = "y_output";
    private TensorFlowInferenceInterface inferenceInterface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        maxTempCheckBox = (CheckBox) findViewById(R.id.max_temp_checkbox);
        minTempCheckBox = (CheckBox) findViewById(R.id.min_temp_checkbox);
        meanTempCheckBox = (CheckBox) findViewById(R.id.mean_temp_checkbox);
        precipCheckBox = (CheckBox) findViewById(R.id.precip_checkbox);
        resultsTextView = (TextView) findViewById(R.id.results_text_view);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    public void predictTemperatureAction(View view) {
        float maxTempGreater = (maxTempCheckBox.isChecked()) ? 1 : 0;
        float minTempGreater = (minTempCheckBox.isChecked()) ? 1 : 0;
        float meanTempGreater = (meanTempCheckBox.isChecked()) ? 1 : 0;
        float precipGreater = (precipCheckBox.isChecked()) ? 1 : 0;
        float[] input = {maxTempGreater, minTempGreater, meanTempGreater, precipGreater};
        float[] results = runInference(input);
        displayResults(results);
    }

    private float[] runInference(float[] input) {
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE, input);
        inferenceInterface.runInference(new String[]{OUTPUT_NODE});
        float[] results = new float[2];
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        return results;
    }

    private void displayResults(float[] results) {
        if (results[0] >= results[1]) {
            resultsTextView.setText("Model predicts temperature will increase");
        } else {
            resultsTextView.setText("Model predicts temperature will decrease");
        }
    }

}
