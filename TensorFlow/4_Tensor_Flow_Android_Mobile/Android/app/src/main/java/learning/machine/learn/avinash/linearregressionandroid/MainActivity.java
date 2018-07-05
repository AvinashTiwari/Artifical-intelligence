package learning.machine.learn.avinash.linearregressionandroid;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_NAME ="file:///android_asset/optimized_frozen_linear_regression.pb";
    private static final String INPUT_NODE ="x";
    private static final String OUTPUT_NODE ="y_output";
    private static final int[] INPUTSHAPE_NODE ={1,1};
    private TensorFlowInferenceInterface InferenceInterface;

    EditText edittext;
    TextView textview;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        edittext = (EditText)findViewById(R.id.edit_text);
        textview = (TextView) findViewById(R.id.text_view);
        InferenceInterface = new TensorFlowInferenceInterface();
        InferenceInterface.initializeTensorFlow(getAssets(), MODEL_NAME);
    }

    public void pressButton(View view){
        float input =  Float.parseFloat(edittext.getText().toString());
       String result =  performInferance(input);
       textview.setText(result);
    }

    private String performInferance(float input){
        float[]  floatArray = {input};
        InferenceInterface.fillNodeFloat(INPUT_NODE,INPUTSHAPE_NODE,floatArray);
        InferenceInterface.runInference(new String[]{OUTPUT_NODE});
        float[]  result = {0.0f};
        InferenceInterface.readNodeFloat(OUTPUT_NODE, result);
        String finalResult =  String.valueOf(result[0]);
        return finalResult;
    }
}
