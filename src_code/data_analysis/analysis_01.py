import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def perform_poison():
    analysis_df = pd.read_csv("storage/game_data.csv", index_col=None)


    # Organize data
    X = analysis_df[['home', 'team_rest', 'opp_rest',
                     'team_avg_goals_for_20',
                     'team_avg_goals_against_20',
                     'opp_avg_goals_for_20',
                     'opp_avg_goals_against_20']] # Independent variables
    y = analysis_df['team_score']  # Dependent variable

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit Poisson regression model on the training data
    poisson_model = PoissonRegressor()
    poisson_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = poisson_model.predict(X_test)
    y_pred_rounded = y_pred.round().astype(int)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_rounded)
    print("Mean Squared Error:", mse)

    r2 = r2_score(y_test, y_pred_rounded)
    print("R-squared (R²):", r2)


def perform_tf():
    analysis_df = pd.read_csv("storage/game_data.csv", index_col=None)
    predict_df = pd.read_csv("storage/future_games.csv", index_col=None)

    keyword_columns = ["team_avg_goals_for_", "team_avg_goals_against_", "opp_avg_goals_for_", "opp_avg_goals_against_"]
    columns_to_extract = ['home', 'team_rest', 'opp_rest']
    for keyword in keyword_columns:
        columns_to_extract.extend(analysis_df.filter(like=keyword).columns.tolist())

    print(f"Columns to extract: {columns_to_extract}")
    # Organize data
    X = analysis_df[columns_to_extract] # Independent variables
    y = analysis_df['non_shootout_win']  # Dependent variable

    X_pred = predict_df[columns_to_extract] # Independent variables

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_pred_normalized = scaler.transform(X_pred)

    # Build the Poisson regression model using TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_normalized.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(16, activation='relu'),  # Second hidden layer with ReLU activation
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_normalized, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test_normalized, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    predictions = model.predict(X_test_normalized)
    future_pred = model.predict(X_pred_normalized)

    # Review predictions
    for i in range(15):
        additional_data = analysis_df.iloc[X_test.index[i]]  # Get additional data from analysis_df corresponding to the current row in X_test
        print(f"Data: {additional_data} \nActual: {y_test.iloc[i]} \tPredicted:  {predictions[i][0]}")

    predict_df['prediction'] = None
    for i in range(len(future_pred)):
        additional_data = predict_df.iloc[X_pred.index[i]]  # Get additional data from analysis_df corresponding to the current row in X_test
        predict_df.loc[X_pred.index[i], 'prediction'] = future_pred[i][0]
        # print(f"Data: {additional_data} \nPredicted:  {future_pred[i][0]}")

    predict_df.to_csv(f"storage/future_games.csv", index=False)