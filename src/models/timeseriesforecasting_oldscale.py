# ancienne méthode de normalisation

X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indexes, test_size=0.33, shuffle = True, random_state = 42)

anomalies_true_test = anomalies_true.loc[index_test]
#anomalies_thresold_test = anomalies_thresold.loc[index_test]
del X
del y

# normaliser: par paramètre. 
# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train_scale = np.reshape(X_train, (len(X_train), nb_jours, n_features))
X_test_scale  = np.reshape(X_test, (len(X_train), nb_jours, n_features))
for i, param in enumerate(parametres):
    # chaque paramètre a son scaler
    scaler_param = MinMaxScaler()
    #on tient compte de la première valeur pour le fit pour éviter de comptabiliser plusieurs fois la même valeur
    scaler_param.fit(X_train_scale[:, 0, i].reshape(-1, 1))
    for j in range(X_train_scale.shape[0]):
        X_train_scale[j, :, i] = scaler_param.transform(X_train_scale[j, :, i].reshape(-1, 1)).reshape(nb_jours)
    for j in range(X_test_scale.shape[0]):
        X_test_scale[j, :, i] = scaler_param.transform(X_test_scale[j, :, i].reshape(-1, 1)).reshape(nb_jours)

y_train = np.array(y_train)
y_test = np.array(y_test)
