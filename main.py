# Libraries needed for linear regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# This function takes a .csv file, separates the x and y values and cleans the data by converting non-numeric values
# to encoded numeric values.
def load_data(filename):

    # reads .csv file
    data = pd.read_csv(filename)

    # creates series (Python's form of an array) for y values.
    sale_price = data['SalePrice']

    # removes the y values, creating a DataFrame (Python's form of a 2-D array) for the X values.
    evidence = data.drop('SalePrice', 1)

    # creates an encoder object which is used to convert non-numeric features to numeric.
    enc = LabelEncoder()

    # convert non-numeric evidence to numeric by looking at each element and if non-numeric, convert, else skip the
    # elements.
    # Looking in the DataFrame of X values.
    for col in evidence:
        # Looking in each element in column.
        for i in range(len(evidence[col])):
            # If non-numeric, encode the elements, which ranks them accordingly from 0 to n-1, where n is number of
            # possible options for specified column (feature).
            if isinstance(evidence[col][i], str) and evidence[col][i] != 'NA':
                enc.fit(evidence[col])
                evidence[col] = enc.transform(evidence[col])
                break

    # Convert NA values to 0, completing the transformation of data from a combination of numeric and non-numeric
    # attributes to a DataFrame of only numbers.
    evidence = evidence.fillna(0)

    # Return the clean DataFrame (evidence) and the series containing the corresponding y-values (sale_price)
    return evidence, sale_price

# ----------------------------------------------------------------------------------------------------------------------


# This function takes the two sets of data that were cleaned by load_data(), evidence and sale_price (labels)
def train_model(evidence, labels):

    # creates a linear regression object, meaning a "line of best fit" was created using all 80 features, with 80
    # coefficients that allow the model to follow the optimal regressive line of the data.
    # In this step, sklearn uses gradient descent to find the 80 parameters.
    model = LinearRegression().fit(evidence, labels)

    # the function returns our object we can use to predict other prices of houses based on its 80 features
    return model

# ----------------------------------------------------------------------------------------------------------------------


# This function will be called multiple times in the function avg_r_value.
def test_model(model, x_test, y_test):

    # Returns the R score of the linear regression model.
    return model.score(x_test, y_test)

# ----------------------------------------------------------------------------------------------------------------------


# This function creates 100 models, finds each r score, averages them, and returns the average r score.
def avg_r_value(file):

    # We will create 100 models to observe.
    sample_size = 100
    # A running total of r scores to be divided by number of models.
    sum_r = 0
    # A running total to account for the unwanted negative r score values.
    error_count = 0

    # Run this operation ~100 times, accounting for errors.
    for i in range(sample_size):

        # Load data from spreadsheet, call method to split into x,y and clean.
        evidence, labels = load_data(file)

        # Split data into test and train sets by using 20% of the data as a test set and the remaining as the train.
        x_train, x_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)

        # Call our function we made that returns a model. We use our newly acquired training sets to do so.
        model = train_model(x_train, y_train)

        # Call our function we made that returns an r score, using the model and our test sets.
        r = test_model(model, x_test, y_test)

        # if-else used here to handle an error that occurs when the r score is negative.
        if r > 0:
            sum_r += r
        else:
            error_count += 1

        # A tacky t-minus counter that shows our program running.
        print('t-', (100-i))
        # END FOR LOOP

    # creates mean r score, accounting for errors.
    r_avg = sum_r/(sample_size - error_count)

    # Prints results of model.
    print("\n\nAverage R value: ", r_avg)
    print('Number of errors (negative r scores): ', error_count)

# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- MAIN CODE --------------------------------------------------------


# CONSTANTS
file_csv = 'train.csv'
TEST_SIZE = 0.2

# METHODS
avg_r_value(file_csv)

# Runtime = ~ 50 seconds


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
