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

    private static final String MODEL_NAME ="file:///android_assets/optimized_frozen_linear_regression.pb";
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
    }

    public void pressButton(View view){

    }
}
