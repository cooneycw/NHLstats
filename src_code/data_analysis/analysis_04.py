import pandas as pd
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


def perform_tf_player(segs):
    analysis_df = pd.read_csv("storage/playerStatsData.csv", index_col=None)
    predict_df = pd.read_csv("storage/playerFutureStatsData.csv", index_col=None)
    if segs is True:
        segs_df = pd.read_csv("storage/segmentation.csv", index_col=None)
        segs_df = segs_df[['player', 'cluster']]
        analysis_df = pd.merge(analysis_df, segs_df, on='player', how='left')
        predict_df = pd.merge(predict_df, segs_df, on='player', how='left')
        special_cluster_value = 99  # You can choose any special value
        analysis_df['cluster'] = analysis_df['cluster'].fillna(special_cluster_value)
        predict_df['cluster'] = predict_df['cluster'].fillna(special_cluster_value)
        analysis_dummies = pd.get_dummies(analysis_df['cluster'], prefix='seg_vals', dtype=int)
        predict_dummies = pd.get_dummies(predict_df['cluster'], prefix='seg_vals', dtype=int)
        col_names = []
        for col_name in analysis_dummies.columns:
            col_names.append(col_name[0:len(col_name)-2])
        analysis_dummies.columns = col_names
        analysis_df = pd.concat([analysis_df, analysis_dummies], axis=1)

        col_names = []
        for col_name in predict_dummies.columns:
            col_names.append(col_name[0:len(col_name)-2])
        predict_dummies.columns = col_names
        predict_df = pd.concat([predict_df, predict_dummies], axis=1)

    columns_to_extract = ['home', 'team_rest', 'opp_rest',
                          'positionCentre', 'positionRightWing', 'positionLeftWing', 'positionDefense',
                          ]

    keyword_columns = ['avg_gamesPlayed_',
                       'avg_gameToi_',
                       'avg_goals_',
                       'avg_anyGoals_',
                       'avg_assists_',
                       'avg_gamePoints_',
                       'avg_anyPoints_',
                       'avg_anyShots_02p_10',
                       'avg_anyShots_03p_10',
                       'avg_anyShots_04p_10',
                       'opp_avg_goals_against_',
                       'opp_avg_shots_against_',
                       ]

    if segs is True:
        keyword_columns = keyword_columns + ['seg_vals_']

    for keyword in keyword_columns:
        columns_to_extract.extend(analysis_df.filter(like=keyword).columns.tolist())

    analysis_cols = ['anyGoals', 'anyPoints', 'anyShots_01p', 'anyShots_02p', 'anyShots_03p', 'anyShots_04p']

    for col in analysis_cols:
        print(f"Columns to extract: {columns_to_extract}")
        # Organize data
        X = analysis_df[columns_to_extract]  # Independent variables
        y = analysis_df[col]

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

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_normalized.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation='sigmoid')
            # Output layer with sigmoid activation for binary classification
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train_normalized, y_train, epochs=100, batch_size=32, validation_split=0.2)

        explainer = shap.Explainer(model, X_train)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_test)

        # Visualize the SHAP values
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"{col} - SHAP Summary Plot")
        plt.show()

        loss, accuracy = model.evaluate(X_test_normalized, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)
        y_pred = model.predict(X_test_normalized)
        future_pred = model.predict(X_pred_normalized)

        predict_df[f'prediction_{col}'] = None
        predict_df[f'{col}_over'] = None
        predict_df[f'{col}_under'] = None
        for i in range(len(future_pred)):
            additional_data = predict_df.iloc[X.index[i]]  # Get additional data from analysis_df corresponding to the current row in X_test
            predict_df.loc[X.index[i], f'prediction_{col}'] = future_pred[i][0]
            predict_df.loc[X.index[i], f'{col}_over'] = round(100 / future_pred[i][0])
            predict_df.loc[X.index[i], f'{col}_under'] = round(100 / (1 - future_pred[i][0]))
            # print(f"Data: {additional_data} \nPredicted:  {future_pred[i][0]}")

        predict_df.to_csv(f"storage/playerFutureStatsData.csv", index=False)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 0])
        roc_auc = roc_auc_score(y_test, y_pred[:, 0])

        # Print performance metrics
        print(f"{col} Accuracy: {accuracy}")
        print(f"{col} Precision: {precision}")
        print(f"{col} Recall: {recall}")
        print(f"{col} F1-score: {f1}")
        print(f"{col} AUC: {roc_auc}")

        # Plot ROC curve

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{col} ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{col} - Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
