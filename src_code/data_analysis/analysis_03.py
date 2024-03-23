import pandas as pd
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.cluster import KMeans


def perform_logistic_player():
    analysis_df = pd.read_csv("storage/playerStatsData.csv", index_col=None)

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

    for keyword in keyword_columns:
        columns_to_extract.extend(analysis_df.filter(like=keyword).columns.tolist())

    analysis_cols = ['anyGoals', 'anyPoints', 'anyShots_01p', 'anyShots_02p', 'anyShots_03p', 'anyShots_04p']

    for col in analysis_cols:
        predict_df = pd.read_csv("storage/playerFutureStatsData.csv", index_col=None)
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

        # Fit logistic regression model
        log_reg_model = LogisticRegression(penalty='elasticnet', l1_ratio=0.75, C=0.25, solver='saga')
        log_reg_model.fit(X_train_normalized, y_train)

        explainer = shap.Explainer(log_reg_model, X_train)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_test)

        # Visualize the SHAP values
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"{col} - SHAP Summary Plot")
        plt.show()

        # Get coefficients and corresponding features
        coefficients = log_reg_model.coef_[0]
        feature_names = X.columns

        # Print coefficients with corresponding feature names
        print(f"{col} Interpretable Coefficients:")
        for feature, coef in zip(feature_names, coefficients):
            print(f"{col} - {feature}: {coef}")

        y_pred = log_reg_model.predict(X_test_normalized)
        future_pred = log_reg_model.predict(X_pred_normalized)
        future_pred_proba = log_reg_model.predict_proba(X_pred_normalized)

        # Extract probabilities of positive class (class 1)
        positive_class_proba = future_pred_proba[:, 1]

        predict_df[f'prediction_{col}'] = None
        predict_df[f'{col}_over'] = None
        predict_df[f'{col}_under'] = None
        for i in range(len(future_pred)):
            additional_data = predict_df.iloc[X.index[i]]  # Get additional data from analysis_df corresponding to the current row in X_test
            predict_df.loc[X.index[i], f'prediction_{col}'] = positive_class_proba[i]
            predict_df.loc[X.index[i], f'{col}_over'] = round(100 / positive_class_proba[i])
            predict_df.loc[X.index[i], f'{col}_under'] = round(100 / (1 - positive_class_proba[i]))
            # print(f"Data: {additional_data} \nPredicted:  {future_pred[i][0]}")

        predict_df.to_csv(f"storage/playerFutureStatsData.csv", index=False)

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


def perform_kmeans_player():
    analysis_df = pd.read_csv("storage/playerStatsData.csv", index_col=None)
    analysis_df['positionForward'] = analysis_df['positionCentre'] + analysis_df['positionRightWing'] + analysis_df['positionLeftWing']

    player_sorted = analysis_df.sort_values(by=['player', 'gameDate'])
    segmentation_df = player_sorted.groupby('player').tail(1).copy()

    columns_to_extract = ['positionForward', 'positionDefense']

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
                       ]

    for keyword in keyword_columns:
        extracted_columns = analysis_df.filter(like=keyword).columns.tolist()
        extracted_columns = [col for col in extracted_columns if 'team' not in col]
        columns_to_extract.extend([col for col in extracted_columns if 'opp' not in col])

    print(f"Columns to extract: {columns_to_extract}")
    scaler = StandardScaler()

    X_seg = scaler.fit_transform(segmentation_df[columns_to_extract])

    # Determine the optimal number of clusters (K) using the elbow method
    n_segs = 30
    wcss = []
    for i in range(1, n_segs):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=20)
        kmeans.fit(X_seg)  # Fit on training data
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_segs), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')  # Within cluster sum of squares
    plt.show()

    # Choose the optimal number of clusters (K)
    optimal_k = 11  # Example: You need to choose based on the elbow method plot

    # Perform K-means clustering with optimal K
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=20)
    kmeans.fit(X_seg)

    # Analyze the characteristics of each cluster
    segmentation_df['Cluster'] = kmeans.labels_
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Convert cluster centers back to original scale

    # Profile the players in each cluster
    for i, center in enumerate(cluster_centers):
        print(f"\nCluster {i+1} Profile:")
        print("Center Values:")
        for col, val in zip(columns_to_extract, center):
            print(f"{col}: {val}")
        print("Top Players:")
        select_list_df = segmentation_df[segmentation_df['Cluster'] == i]
        top_players = select_list_df.head(5)  # Adjust the number of top players as needed
        print(top_players[['player', 'playerLastName']])  # Assuming these columns exist in your DataFrame

    segmentation_df.to_csv("storage/segmentation.csv", index=False)
