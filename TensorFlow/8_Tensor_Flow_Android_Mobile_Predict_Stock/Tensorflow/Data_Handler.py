import requests


# Function to retrieve data from Barchart.com's API
# Retrives historical data between start and end dates for the stock symbol that we enter. Response is in JSON format
def get_stock_data(stock_symbol, start_date, end_date):
    base_url = 'https://marketdata.websol.barchart.com/getHistory.json'
    api_key = '02cd306bc324ba16a64eeca25d6f5213'
    params = {'apikey': api_key,  'symbol': stock_symbol, 'type': 'daily', 'startDate': start_date, 'endDate': end_date}
    data_response = requests.get(base_url, params)
    print(data_response.url)
    return data_response.json()


# Function to parse the JSON data and extract the info we need into the 5 categories
def decode_json_response(json_data):
    open_prices = []
    close_prices = []
    highs = []
    lows = []
    volumes = []
    results = json_data.get('results')
    for result in results:
        open_prices.append(result.get('open'))
        close_prices.append(result.get('close'))
        highs.append(result.get('high'))
        lows.append(result.get('low'))
        volumes.append(result.get('volume'))
    return open_prices, close_prices, highs, lows, volumes


# Function to calculate the difference in values between the next data point and the current data point
# Store a 1 if the price goes up the next day and 0 if the price goes down
def calculate_differences(input_array):
    differences = []
    for i in range(len(input_array) - 1):
        difference = input_array[i + 1] - input_array[i]
        if difference >= 0:
            differences.append(1)
        else:
            differences.append(0)
    return differences


# Function to assign a label based on the difference between open and close prices
# If the open price the next day is higher than the close price of the previous day store [1 0] otherwise store [0 1]
def calculate_price_difference(open_prices, close_prices):
    price_differences = []
    for i in range(len(open_prices) - 1):
        difference = open_prices[i + 1] - close_prices[i]
        if difference >= 0:
            price_differences.append([1, 0])
        else:
            price_differences.append([0, 1])
    return price_differences


# Function to build the proper data sets for training and testing
# Fetches and formats the data, builds the sets for feed into the model, and assigns labels
def build_data_subsets(stock_symbol, start_date, end_date):
    json_data = get_stock_data(stock_symbol, start_date, end_date)
    open_prices, close_prices, highs, lows, volumes = decode_json_response(json_data)

    open_diffs = calculate_differences(open_prices)
    close_diffs = calculate_differences(close_prices)
    high_diffs = calculate_differences(highs)
    low_diffs = calculate_differences(lows)
    volume_diffs = calculate_differences(volumes)

    labels = calculate_price_difference(open_prices, close_prices)

    final_data = []
    for i in range(len(open_diffs)):
        final_data.append([open_diffs[i], close_diffs[i], high_diffs[i], low_diffs[i], volume_diffs[i]])

    return final_data, labels
