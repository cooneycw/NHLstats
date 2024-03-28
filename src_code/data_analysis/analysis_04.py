import pandas as pd
import numpy as np
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
        model_col_names = []
        for col_name in analysis_dummies.columns:
            model_col_names.append(col_name[0:len(col_name)-2])
        analysis_dummies.columns = model_col_names
        analysis_df = pd.concat([analysis_df, analysis_dummies], axis=1)

        col_names = []
        for col_name in predict_dummies.columns:
            col_names.append(col_name[0:len(col_name)-2])
        predict_dummies.columns = col_names
        predict_dummies = predict_dummies.reindex(columns=model_col_names, fill_value=0)
        predict_df = pd.concat([predict_df, predict_dummies], axis=1)

    columns_to_extract = ['home', 'team_rest', 'opp_rest',
                          'positionCentre', 'positionRightWing', 'positionLeftWing', 'positionDefense',
                          ]

    keyword_columns = ['avg_gamesPlayed_',
                       'avg_toi_',
                       'avg_pim_',
                       'avg_hits_',
                       'avg_goals_',
                       'avg_anyGoals_',
                       'avg_assists_',
                       'avg_anyAssists_',
                       'avg_points_',
                       'avg_anyPoints_',
                       'avg_shots_',
                       'avg_anyShots_01p_',
                       'avg_anyShots_02p_',
                       'avg_anyShots_03p_',
                       'avg_anyShots_04p_',
                       'opp_avg_goals_against_',
                       'opp_avg_shots_against_',
                       'opp_avg_goals_for_',
                       'opp_avg_shots_for_',
                       'opp_avg_hits_for_',
                       'opp_avg_blocks_for_',
                       'opp_avg_pim_for_',
                       ]

    if segs is True:
        keyword_columns = keyword_columns + ['seg_vals_']

    for keyword in keyword_columns:
        columns_to_extract.extend(analysis_df.filter(like=keyword).columns.tolist())

    analysis_cols = ['anyGoals', 'anyAssists',  'anyPoints', 'anyShots_01p', 'anyShots_02p', 'anyShots_03p', 'anyShots_04p']

    predictions = []
    for col in analysis_cols:
        print(f"Modelling: {col} - Columns to extract: {columns_to_extract}")
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

        shape = X_train_normalized.shape
        l2_lambda = 0.001
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_normalized.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid')
            # Output layer with sigmoid activation for binary classification
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train_normalized, y_train, epochs=5, batch_size=18, validation_split=0.2)

        # Access training history
        training_loss = history.history['loss']
        training_accuracy = history.history['accuracy']
        validation_loss = history.history['val_loss']
        validation_accuracy = history.history['val_accuracy']

        # Plot training curves
        epochs = range(1, len(training_loss) + 1)

        plt.plot(epochs, training_loss, 'bo', label='Training loss')
        plt.plot(epochs, validation_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(epochs, training_accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, validation_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        loss, accuracy = model.evaluate(X_test_normalized, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)
        future_pred = model.predict(X_pred_normalized)

        target_cols = []
        target_cols.extend(predict_df.filter(like=col).columns.tolist())

        prediction_data = {
            'player': predict_df['player'],  # Assuming 'player' column exists in predict_df
        }
        for target_col in target_cols:
            prediction_data[target_col] = np.round(100 * predict_df[target_col], 1)

        prediction_data[f'prediction_{col}'] = np.round(100 * future_pred[:, 0], 1)
        prediction_data[f'{col}_over'] = np.round(100 / future_pred[:, 0])
        prediction_data[f'{col}_under'] = np.round(100 / (1 - future_pred[:, 0]))

        # Append predictions for current column to the list
        predictions.append(pd.DataFrame(prediction_data))

    predictions_df = pd.concat(predictions, axis=1)
    final_df = pd.concat([predict_df, predictions_df], axis=1)

    final_df.to_excel("storage/playerFuturePredTfData.xlsx", index=False)
