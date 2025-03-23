from api.data_preprocessing import normalize_data
TARGETS = ["thermique", "nucleaire", "eolien", "solaire", "hydraulique", "bioenergies"]


def get_predictions_per_target_dict(model, X_test, df, features, n_days=50):
    """
    Renvoie un dictionnaire contenant les prédictions du modèle pour chaque cible sur n jours (par pas de 3h).

    Args:
        model: modèle entraîné (LSTM ou Bi-LSTM)
        X_test: séquences de test
        df: DataFrame non normalisé (pour obtenir le scaler_y depuis normalize_data)
        features: liste des features
        n_days: nombre de jours à prédire (1 jour = 8 pas pour 3h)

    Returns:
        dict: {target_name: [val1, val2, ..., valN]} sur 50 jours
    """
    _, _, scaler_y = normalize_data(df.copy(), features)
    n_steps = n_days * 8
    #y_pred = model.predict(X_test, verbose=0)
    y_pred = model.predict(X_test[:, :, :-11], verbose=0)
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_pred_real = y_pred_real[:n_steps]
    prediction_dict = {target: y_pred_real[:, i].tolist() for i, target in enumerate(TARGETS)}
    return prediction_dict


def get_real_values_per_target_dict(y_test, df, features, n_days=50):
    """
    Renvoie un dictionnaire des valeurs réelles inversées sur les 50 jours.

    Args:
        y_test: tableau des valeurs cibles normalisées
        df: DataFrame non normalisé (pour obtenir le scaler_y depuis normalize_data)
        features: liste des features
        n_days: nombre de jours à afficher (1 jour = 8 pas pour 3h)

    Returns:
        dict: {target_name: [val1, val2, ..., valN]}
    """
    _, _, scaler_y = normalize_data(df.copy(), features)
    n_steps = n_days * 8
    y_real = scaler_y.inverse_transform(y_test[:n_steps])
    real_dict = {target: y_real[:, i].tolist() for i, target in enumerate(TARGETS)}
    return real_dict


def format_predictions_json(prediction_dict):
    """
    Transforme le dictionnaire de prédictions en un format JSON-friendly (liste de dicts).

    Args:
        prediction_dict: dictionnaire {target_name: [val1, val2, ...]}

    Returns:
        list: [{"step": int, "target_1": float, ..., "target_n": float}]
    """
    steps = len(next(iter(prediction_dict.values())))
    formatted = []
    for i in range(steps):
        row = {target: prediction_dict[target][i] for target in prediction_dict}
        row["step"] = i
        formatted.append(row)
    return formatted
