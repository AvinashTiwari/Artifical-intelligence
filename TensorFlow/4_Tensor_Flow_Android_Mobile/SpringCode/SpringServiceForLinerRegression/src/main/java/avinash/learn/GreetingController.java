package avinash.learn;

import java.util.concurrent.atomic.AtomicLong;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import android.app.AliasActivity;

@RestController
public class GreetingController extends AliasActivity {
	
	
    private static final String MODEL_NAME ="F:/AI/TensorFlow/4_Tensor_Flow_Android_Mobile/SpringCode/SpringServiceForLinerRegression/src/main/resources/optimized_frozen_linear_regression.pb";
    private static final String INPUT_NODE ="x";
    private static final String OUTPUT_NODE ="y_output";
    private static final int[] INPUTSHAPE_NODE ={1,1};
    private TensorFlowInferenceInterface InferenceInterface;
	
	@RequestMapping("/greeting")
    public String greeting(@RequestParam(value="name", defaultValue="World") String name) {
        InferenceInterface = new TensorFlowInferenceInterface();
        InferenceInterface.initializeTensorFlow(getAssets(), MODEL_NAME);
        
        return "Hello " + performInferance(12);
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
