package android.learn.avinash.titanicsurvier;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/frozen_Titanic_prediction.pb";
    private static final String INPUT_NODE = "x_input";
    private static final int[] INPUT_SHAPE = {1, 6};
    private static final String OUTPUT_NODE = "y_output";
    private TensorFlowInferenceInterface inferenceInterface;

    TextView resultsTextView;
    EditText passangerClass;
    EditText gender;
    EditText age;
    EditText sibling;
    EditText fare;
    EditText parcel;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);

        resultsTextView = (TextView) findViewById(R.id.Result);


         passangerClass = (EditText) findViewById(R.id.passangerClass);
         gender = (EditText) findViewById(R.id.gender);
         age = (EditText) findViewById(R.id.age);;
         sibling =(EditText) findViewById(R.id.sibling);
         fare =(EditText) findViewById(R.id.fare);
         parcel=(EditText) findViewById(R.id.parcel);

    }


    private float[] runInference(float[] input) {
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE, input);
        inferenceInterface.runInference(new String[]{OUTPUT_NODE});
        float[] results = new float[2];
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        return results;
    }


    public void predcitSurving(View view) {
        //Pclass	Sex	Age	SibSp	Parch	Fare
        int Pclass = Integer.parseInt(passangerClass.getText().toString());
        int Sex = Integer.parseInt(gender.getText().toString());
        int Age = Integer.parseInt(age.getText().toString());
        int SibSp =Integer.parseInt(sibling.getText().toString());
        int Parch =Integer.parseInt(parcel.getText().toString());
        int Fare = Integer.parseInt(fare.getText().toString());


        float[] input = {Pclass,Sex,Age,SibSp,Parch,Fare};
        float[] results = runInference(input);
        displayResults(results);
    }


    private void displayResults(float[] results) {
        if (results[0] >= results[1]) {
            resultsTextView.setText("Model predicts passenger  will Survive");
        } else {
            resultsTextView.setText("Model predicts passenger will no  Survive");
        }
    }
}
