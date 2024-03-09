import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def perform_tf():
    analysis_df = pd.read_csv("storage/game_data.csv", index_col=None)
    analysis_df['wins'] = analysis_df['non_shootout_win'] + 0 * analysis_df['shootout_win'] # Dependent variable
    predict_df = pd.read_csv("storage/future_games.csv", index_col=None)

    keyword_columns = ["team_avg_wins_", "opp_avg_wins_",
                       "team_avg_goals_for_", "team_avg_goals_against_",
                       "opp_avg_goals_for_", "opp_avg_goals_against_",
                       "team_avg_shots_for_", "team_avg_shots_against_",
                       "opp_avg_shots_for_", "opp_avg_shots_against_",
                       "team_avg_pim_for_", "team_avg_pim_against_",
                       "opp_avg_pim_for_", "opp_avg_pim_against_",
                       "team_avg_blocks_for_", "team_avg_blocks_against_",
                       "opp_avg_blocks_for_", "opp_avg_blocks_against_",
                       "team_avg_hits_for_", "team_avg_hits_against_",
                       "opp_avg_hits_for_", "opp_avg_hits_against_",
                       "team_avg_face_for_", "team_avg_face_against_",
                       "opp_avg_face_for_", "opp_avg_face_against_",
                       ]
    columns_to_extract = ['home', 'team_rest', 'opp_rest']
    for keyword in keyword_columns:
        columns_to_extract.extend(analysis_df.filter(like=keyword).columns.tolist())

    print(f"Columns to extract: {columns_to_extract}")
    # Organize data
    X = analysis_df[columns_to_extract] # Independent variables
    y = analysis_df['wins']

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
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_normalized, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # xgbmodel = XGBClassifier()
    #
    # # Train the model
    # xgbmodel.fit(X_train, y_train)
    #
    # # Make predictions on the test set
    # y_pred = xgbmodel.predict(X_test)
    #
    # # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print("XGB Accuracy:", accuracy)

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