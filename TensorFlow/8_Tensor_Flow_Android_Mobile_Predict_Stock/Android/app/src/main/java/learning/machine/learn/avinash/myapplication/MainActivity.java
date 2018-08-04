package learning.machine.learn.avinash.myapplication;

import android.app.ProgressDialog;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public class MainActivity extends AppCompatActivity {

    EditText stockSymbolEditText, startDateEditText, endDateEditText;
    TextView resultsTextView;

    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/optimized_stock_prediction.pb";
    private static final String INPUT_NODE = "x_input";
    private static final int[] INPUT_SHAPE = {1, 5};
    private static final String OUTPUT_NODE = "y_output";
    private TensorFlowInferenceInterface inferenceInterface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        stockSymbolEditText = (EditText) findViewById(R.id.stock_symbol_edit_text);
        startDateEditText = (EditText) findViewById(R.id.start_date_edit_text);
        endDateEditText = (EditText) findViewById(R.id.end_date_edit_text);
        resultsTextView = (TextView) findViewById(R.id.text_view);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    public void predictAction(View view) {
        String stockSymbol = stockSymbolEditText.getText().toString();
        String startDate = startDateEditText.getText().toString();
        String endDate = endDateEditText.getText().toString();

        String urlString = "https://marketdata.websol.barchart.com/getHistory.json?" +
                "apikey=3843ebbe81a003184ce8b6662d7b5967&symbol=" +
                stockSymbol + "&type=daily&startDate=" +
                startDate + "&endDate=" + endDate;
        new JsonTask().execute(urlString);
    }

    private void parseJSONData(String jsonString) {
        try {
            float open;
            float close;
            float high;
            float low;
            float volume;

            JSONObject rootObject = new JSONObject(jsonString);
            JSONArray results = rootObject.getJSONArray("results");
            JSONObject todaysData = results.getJSONObject(results.length() - 1);
            JSONObject yesterdaysData = results.getJSONObject(results.length() - 2);
            open = (float) (todaysData.getDouble("open") - yesterdaysData.getDouble("open"));
            open = (open >= 0) ? 1 : 0;
            close = (float) (todaysData.getDouble("close") - yesterdaysData.getDouble("close"));
            close = (close >= 0) ? 1 : 0;
            high = (float) (todaysData.getDouble("high") - yesterdaysData.getDouble("high"));
            high = (high >= 0) ? 1 : 0;
            low = (float) (todaysData.getDouble("low") - yesterdaysData.getDouble("low"));
            low = (low >= 0) ? 1 : 0;
            volume = (float) (todaysData.getDouble("volume") - yesterdaysData.getDouble("volume"));
            volume = (volume >= 0) ? 1 : 0;

            float[] input = {open, close, high, low, volume};
            runInference(input);

        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    private void runInference(float[] input) {
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE, input);
        inferenceInterface.runInference(new String[]{OUTPUT_NODE});
        float[] results = new float[2];
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        displayResults(results);
    }

    private void displayResults(float[] results) {
        if (results[0] >= results[1]) {
            resultsTextView.setText("Model predicts price will increase");
        } else {
            resultsTextView.setText("Model predicts price will decrease");
        }
    }

    private class JsonTask extends AsyncTask<String, String, String> {

        ProgressDialog pd;

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            pd = new ProgressDialog(MainActivity.this);
            pd.setMessage("Please wait");
            pd.setCancelable(false);
            pd.show();
        }

        @Override
        protected String doInBackground(String... params) {

            HttpURLConnection connection = null;
            BufferedReader reader = null;

            try {

                URL url = new URL(params[0]);
                connection = (HttpURLConnection) url.openConnection();
                connection.connect();

                InputStream stream = connection.getInputStream();
                reader = new BufferedReader(new InputStreamReader(stream));
                StringBuffer buffer = new StringBuffer();
                String line = "";

                while((line = reader.readLine()) != null) {
                    buffer.append(line+ "\n");
                }
                return buffer.toString();

            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (connection != null) {
                    connection.disconnect();
                }
                try {
                    if (reader != null) {
                        reader.close();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return null;
        }

        @Override
        protected void onPostExecute(String s) {
            super.onPostExecute(s);
            if (pd.isShowing()) {
                pd.dismiss();
            }
            parseJSONData(s);
        }
    }
}
