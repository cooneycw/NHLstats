import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


def perform_logistic_goals(goal_test):
    analysis_df = pd.read_csv("storage/game_data.csv", index_col=None)
    analysis_df['team_opp_rest'] = analysis_df['team_rest'] * analysis_df['opp_rest']
    analysis_df['over_under'] = (analysis_df['team_score'] + analysis_df['opp_score'] > goal_test).astype(int)
    predict_df = pd.read_csv("storage/future_games.csv", index_col=None)
    predict_df['team_opp_rest'] = predict_df['team_rest'] * predict_df['opp_rest']

    keyword_columns = ["team_avg_wins_", "opp_avg_wins_",
                       "team_avg_goals_for_", "team_avg_goals_against_",
                       "opp_avg_goals_for_", "opp_avg_goals_against_",
                       "team_avg_shots_for_", "team_avg_shots_against_",
                       "team_avg_pp_for_", "team_avg_pp_against_",
                       "team_avg_give_for_", "team_avg_give_against_",
                       "team_avg_take_for_", "team_avg_take_against_",
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
    columns_to_extract = ['home', 'team_rest', 'opp_rest', 'team_opp_rest']
    for keyword in keyword_columns:
        columns_to_extract.extend(analysis_df.filter(like=keyword).columns.tolist())

    print(f"Columns to extract: {columns_to_extract}")
    # Organize data
    X = analysis_df[columns_to_extract]  # Independent variables
    y = analysis_df['over_under']

    X_pred = predict_df[columns_to_extract]  # Independent variables

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_pred_normalized = scaler.transform(X_pred)

    # Fit logistic regression model
    log_reg_model = LogisticRegression(penalty='l1', C=0.5, solver='liblinear')
    log_reg_model.fit(X_train_normalized, y_train)

    # Get coefficients and corresponding features
    coefficients = log_reg_model.coef_[0]
    feature_names = X.columns

    # Print coefficients with corresponding feature names
    print("Interpretable Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef}")

    y_pred = log_reg_model.predict(X_test_normalized)
    future_pred = log_reg_model.predict(X_pred_normalized)
    future_pred_proba = log_reg_model.predict_proba(X_pred_normalized)

    # Extract probabilities of positive class (class 1)
    positive_class_proba = future_pred_proba[:, 1]

    predict_df['prediction'] = None
    for i in range(len(future_pred)):
        additional_data = predict_df.iloc[X_pred.index[i]]  # Get additional data from analysis_df corresponding to the current row in X_test
        predict_df.loc[X_pred.index[i], 'prediction'] = positive_class_proba[i]
        # print(f"Data: {additional_data} \nPredicted:  {future_pred[i][0]}")

    predict_df.to_csv(f"storage/future_games_over_{goal_test}.csv", index=False)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate ROC curve and AUC
    y_proba = log_reg_model.predict_proba(X_test_normalized)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print performance metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"AUC: {roc_auc}")

    # Plot ROC curve

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def perform_logistic():
    analysis_df = pd.read_csv("storage/game_data.csv", index_col=None)
    analysis_df['team_opp_rest'] = analysis_df['team_rest'] * analysis_df['opp_rest']
    analysis_df['wins'] = analysis_df['non_shootout_win'] + 1 * analysis_df['shootout_win']  # Dependent variable
    predict_df = pd.read_csv("storage/future_games.csv", index_col=None)
    predict_df['team_opp_rest'] = predict_df['team_rest'] * predict_df['opp_rest']

    keyword_columns = ["team_avg_wins_", "opp_avg_wins_",
                       "team_avg_goals_for_", "team_avg_goals_against_",
                       "opp_avg_goals_for_", "opp_avg_goals_against_",
                       "team_avg_shots_for_", "team_avg_shots_against_",
                       "team_avg_pp_for_", "team_avg_pp_against_",
                       "team_avg_give_for_", "team_avg_give_against_",
                       "team_avg_take_for_", "team_avg_take_against_",
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
    columns_to_extract = ['home', 'team_rest', 'opp_rest', 'team_opp_rest']
    for keyword in keyword_columns:
        columns_to_extract.extend(analysis_df.filter(like=keyword).columns.tolist())

    print(f"Columns to extract: {columns_to_extract}")
    # Organize data
    X = analysis_df[columns_to_extract]  # Independent variables
    y = analysis_df['wins']

    # poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # X_interactions = poly.fit_transform(X)
    # feature_names = poly.get_feature_names_out(X.columns)

    # Create a new dataframe with these new features to see the column names clearly
    # X_interactions = pd.DataFrame(X_interactions, columns=feature_names)

    X_pred = predict_df[columns_to_extract]  # Independent variables
    # X_pred_interactions = poly.fit_transform(X_pred)
    # X_pred_interactions = pd.DataFrame(X_pred_interactions, columns=feature_names)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_pred_normalized = scaler.transform(X_pred)

    # Fit logistic regression model
    log_reg_model = LogisticRegression(penalty='elasticnet', l1_ratio=0.75, C=0.25, solver='saga')
    log_reg_model.fit(X_train_normalized, y_train)

    explainer = shap.Explainer(log_reg_model, X_train)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    # Visualize the SHAP values
    shap.summary_plot(shap_values, X_test)

    # Get coefficients and corresponding features
    coefficients = log_reg_model.coef_[0]
    feature_names = X.columns

    # Print coefficients with corresponding feature names
    print("Interpretable Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef}")

    y_pred = log_reg_model.predict(X_test_normalized)
    future_pred = log_reg_model.predict(X_pred_normalized)
    future_pred_proba = log_reg_model.predict_proba(X_pred_normalized)

    # Extract probabilities of positive class (class 1)
    positive_class_proba = future_pred_proba[:, 1]

    predict_df['prediction'] = None
    for i in range(len(future_pred)):
        additional_data = predict_df.iloc[X.index[i]]  # Get additional data from analysis_df corresponding to the current row in X_test
        predict_df.loc[X.index[i], 'prediction'] = positive_class_proba[i]
        # print(f"Data: {additional_data} \nPredicted:  {future_pred[i][0]}")

    predict_df.to_csv(f"storage/future_games.csv", index=False)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate ROC curve and AUC
    y_proba = log_reg_model.predict_proba(X_test_normalized)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Print performance metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"AUC: {roc_auc}")

    # Plot ROC curve

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def perform_tf():
    analysis_df = pd.read_csv("storage/game_data.csv", index_col=None)
    analysis_df['wins'] = analysis_df['non_shootout_win'] + 0 * analysis_df['shootout_win'] # Dependent variable
    analysis_df['over_01p'] = (analysis_df['team_score'] + analysis_df['opp_score'] > 1).astype(int)
    analysis_df['over_02p'] = (analysis_df['team_score'] + analysis_df['opp_score'] > 2).astype(int)
    analysis_df['over_03p'] = (analysis_df['team_score'] + analysis_df['opp_score'] > 3).astype(int)
    analysis_df['over_04p'] = (analysis_df['team_score'] + analysis_df['opp_score'] > 4).astype(int)
    analysis_df['over_05p'] = (analysis_df['team_score'] + analysis_df['opp_score'] > 5).astype(int)
    analysis_df['over_06p'] = (analysis_df['team_score'] + analysis_df['opp_score'] > 6).astype(int)
    analysis_df['over_07p'] = (analysis_df['team_score'] + analysis_df['opp_score'] > 7).astype(int)

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

    predictions = []
    cols = ['wins', 'over_01p', 'over_02p', 'over_03p', 'over_04p', 'over_05p', 'over_06p', 'over_07p']
    for col in cols:
        print(f"Modelling: {col} - Columns to extract: {columns_to_extract}")
        y = analysis_df[col]

        X_pred = predict_df[columns_to_extract]  # Independent variables

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        X_pred_normalized = scaler.transform(X_pred)

        # Build the Poisson regression model using TensorFlow
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_normalized.shape[1],)),
            tf.keras.layers.Dense(32, activation='LeakyReLU'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train_normalized, y_train, epochs=20, batch_size=128, validation_split=0.2)

        loss, accuracy = model.evaluate(X_test_normalized, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        future_pred = model.predict(X_pred_normalized)

        target_cols = []
        target_cols.extend(predict_df.filter(like=col).columns.tolist())

        prediction_data = {
            'game_id': predict_df['game_id'],  # Assuming 'player' column exists in predict_df
        }
        for target_col in target_cols:
            prediction_data[target_col] = np.round(100 * predict_df[target_col], 1)

        prediction_data[f'{col}_prediction'] = np.round(100 * future_pred[:, 0], 1)
        prediction_data[f'{col}_over'] = np.round(100 / future_pred[:, 0])
        prediction_data[f'{col}_under'] = np.round(100 / (1 - future_pred[:, 0]))

        # Append predictions for current column to the list
        predictions.append(pd.DataFrame(prediction_data))

    predictions_df = pd.concat(predictions, axis=1)
    final_df = pd.concat([predict_df, predictions_df], axis=1)

    final_df.to_excel(f"storage/futureTfGames.xlsx", index=False)