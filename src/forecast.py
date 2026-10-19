import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

def prepare_demand_history(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    prepare demand history for forecasting. 
    It calculates rolling averages and standard deviations of the last 7 and 14 days of real units dispatched for each distribution center. 
    It also adds a column indicating when the total for a day is available (the next day).

    Args:
        df_hist (pd.DataFrame): DataFrame containing historical dispatch data with columns 'centro_distribucion', 'fecha_despacho', and 'unidades_reales'.
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for rolling averages, standard deviations, and availability date.
    """

    hist = df_hist.copy()

    hist["fecha_despacho"] = pd.to_datetime(
        hist["fecha_despacho"]
    )

    hist = hist.sort_values(
        ["centro_distribucion", "fecha_despacho"]
    )

    for dias in [7, 14]:
        hist[f"promedio_{dias}_dias"] = (
            hist.groupby("centro_distribucion")["unidades_reales"]
            .transform(
                lambda s: s.rolling(dias, min_periods=dias).mean()
            )
        )

        hist[f"std_{dias}_dias"] = (
            hist.groupby("centro_distribucion")["unidades_reales"]
            .transform(
                lambda s: s.rolling(dias, min_periods=dias).std()
            )
        )

    hist["fecha_disponible"] = (
        hist["fecha_despacho"] + pd.Timedelta(days=1)
    )

    return hist 

def calculate_diff_pred_real_base(df_forecast_base: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the difference between forecasted and real units for each row in the DataFrame.
    Additionally, it computes the rolling averages of these differences over the last 7 and 14 days for each distribution center and forecast horizon.

    Args:
        df_forecast_base (pd.DataFrame): DataFrame containing forecasted and real units with columns 'unidades_pronosticadas' and 'unidades_reales'.
    returns:
        pd.DataFrame: DataFrame with an additional column 'diff_forecast_base_real' representing the difference between forecasted and real units.
    """
    df_forecast_new = df_forecast_base.copy()
    df_forecast_new["fecha_despacho"] = pd.to_datetime(
    df_forecast_new["fecha_despacho"], errors="coerce"
    )
    df_forecast_new["dia_semana_despacho"] = (
        df_forecast_new["fecha_despacho"].dt.dayofweek
    )

    df_forecast_new['diff_forecast_base_real'] = df_forecast_new['unidades_pronosticadas'] - df_forecast_new['unidades_reales']

    grupos_error = ["centro_distribucion", "horizonte"]

    hist_error = df_forecast_new.copy().sort_values(
        ["centro_distribucion", "horizonte", "fecha_despacho"]
    )

    for dias in [7, 14]:
        hist_error[f"promedio_error_{dias}_dias"] = (
            hist_error.groupby(grupos_error)["diff_forecast_base_real"]
            .transform(
                lambda s: s.rolling(dias, min_periods=dias).mean()
            )
        )

    hist_error["fecha_disponible_error"] = (
        hist_error["fecha_despacho"] + pd.Timedelta(days=1)
    )

    df_forecast_new["fecha_emision"] = pd.to_datetime(
        df_forecast_new["fecha_emision"]
    )

    columnas_error = grupos_error + [
        "fecha_disponible_error",
        "promedio_error_7_dias",
        "promedio_error_14_dias",
    ]

    df_forecast_new = df_forecast_new.merge(
        hist_error[columnas_error],
        left_on=grupos_error + ["fecha_emision"],
        right_on=grupos_error + ["fecha_disponible_error"],
        how="left",
        validate="many_to_one",
    ).drop(columns="fecha_disponible_error")

    return df_forecast_new

def prepare_dummies(
    df_forecast_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    
    Encode day, center, and horizon

    Args:
        df_forecast_base (pd.DataFrame): DataFrame containing forecasted and real units.
    returns:
        tuple[pd.DataFrame, list[str]]: forecast with dummies and columns of dummies.
    
    """
    df_forecast_dummies = df_forecast_base.copy()
    categoricas = [
        "dia_semana_despacho",
        "centro_distribucion",
        "horizonte",
    ]

    dummies = pd.get_dummies(
        df_forecast_dummies[categoricas],
        columns=categoricas,
        prefix=["dia", "cd", "horizonte"],
        dtype=int,
    )

    df_forecast_dummies = (
        df_forecast_dummies
        .drop(columns="dia_semana_despacho")
        .join(dummies)
    )

    return df_forecast_dummies, dummies.columns.tolist()

def prepare_forecast_features(
    df_hist: pd.DataFrame,
    df_forecast_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare features for forecasting by combining historical dispatch data with forecasted data.
    It calculates rolling averages and standard deviations of the last 7 and 14 days of real units dispatched for each distribution center.
    Additionally, it computes the difference between forecasted and real units, as well as rolling averages of these differences.

    Args:
        df_hist (pd.DataFrame): DataFrame containing historical dispatch data with columns 'centro_distribucion', 'fecha_despacho', and 'unidades_reales'.
        df_forecast_base (pd.DataFrame): DataFrame containing forecasted data with columns 'centro_distribucion', 'horizonte', 'fecha_emision', and 'unidades_pronosticadas'.

    Returns:
        tuple[pd.DataFrame, list[str]]: Model-ready data and the names of the one-hot encoded columns.

    """
    hist = prepare_demand_history(df_hist)
    df_forecast_new = calculate_diff_pred_real_base(df_forecast_base)

    df_forecast_new["fecha_emision"] = pd.to_datetime(
    df_forecast_new["fecha_emision"]
    )

    columnas = [
        "centro_distribucion",
        "fecha_disponible",
        "promedio_7_dias",
        "promedio_14_dias",
        "std_7_dias",
        "std_14_dias",
    ]

    df_forecast_new = df_forecast_new.merge(
        hist[columnas],
        left_on=["centro_distribucion", "fecha_emision"],
        right_on=["centro_distribucion", "fecha_disponible"],
        how="left",
        validate="many_to_one",
    ).drop(columns="fecha_disponible")


    df_forecast_dummies, columnas_codificadas = prepare_dummies(
        df_forecast_new
    )

    df_modelo = df_forecast_dummies.dropna().copy()
    
    return df_modelo, columnas_codificadas

def select_backtest_dates(df: pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Selects the dates for training, validation, and testing based on the emission dates in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing forecasted data with a 'fecha_emision' column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]: A tuple containing the training DataFrame, validation DataFrame, testing DataFrame, validation emission dates, and testing emission dates.
        
    """
    df_modelo = df.copy()
    fechas = pd.DatetimeIndex(
        df_modelo["fecha_emision"].unique()
    ).sort_values()

    horizonte_maximo = pd.Timedelta(days=4)

    fechas = fechas[
        fechas + horizonte_maximo <= df_modelo["fecha_despacho"].max()
    ]

    fechas_previas, fechas_test = train_test_split(
        fechas,
        test_size=3,
        shuffle=False,
    )

    fechas_previas = fechas_previas[
        fechas_previas + horizonte_maximo < fechas_test.min()
    ]

    _, fechas_val = train_test_split(
        fechas_previas,
        test_size=4,
        shuffle=False,
    )

    train = df_modelo.loc[
        df_modelo["fecha_despacho"] < fechas_val.min()
    ].copy()

    val = df_modelo.loc[
        df_modelo["fecha_emision"].isin(fechas_val)
    ].copy()

    test = df_modelo.loc[
        df_modelo["fecha_emision"].isin(fechas_test)
    ].copy()

    return train, val, test, fechas_val, fechas_test

def create_forecast_pipeline(
    parametros: dict[str, list[float]],
    columnas_numericas: list[str],
    columnas_dummies_codificadas: list[str],
) -> tuple[Pipeline, list[dict[str, float]]]:
    """ 
    Creates a scikit-learn pipeline for forecasting, including preprocessing and model steps.

    Args:
        parametros (dict): Dictionary containing hyperparameters for the model.
        columnas_numericas (list[str]): List of numerical feature column names.
        columnas_dummies_codificadas (list[str]): List of one-hot encoded categorical feature column names.

    Returns:
        tuple[Pipeline, list[dict]]: A tuple containing the scikit-learn pipeline and a list of configuration dictionaries for hyperparameter tuning.
    """

    preprocesamiento = ColumnTransformer(
    transformers=[
        ("numericas", StandardScaler(), columnas_numericas),
        ("codificadas", "passthrough", columnas_dummies_codificadas),
    ]
    )


    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", ElasticNet(max_iter=20000)),
        ]
    )
    configuraciones = list(ParameterGrid(parametros))
    
    return pipeline, configuraciones


def predict_rolling_origin(
    df_modelo: pd.DataFrame,
    fechas: pd.DatetimeIndex,
    pipeline: Pipeline,
    configuracion: dict,
    features: list[str],
    objetivo: str,
) -> pd.DataFrame:
    """
    Performs rolling origin predictions for each emission date in the provided DataFrame.

    Args:
        df_modelo (pd.DataFrame): DataFrame containing the model data with features and target
        fechas (pd.DatetimeIndex): Emission dates for which predictions will be made.
        pipeline (Pipeline): A scikit-learn pipeline containing preprocessing and model steps.
        configuracion (dict): Configuration parameters for the model.
        features (list[str]): List of feature column names to be used for prediction.
        objetivo (str): Name of the target column to be predicted.

    Returns:
        pd.DataFrame: DataFrame containing the original data along with the predicted differences and corrected forecasts for each emission date.
    """

    predicciones_folds = []

    for fecha_origen in fechas.sort_values():

        train_fold = df_modelo.loc[
            df_modelo["fecha_despacho"] < fecha_origen
        ].copy()

        evaluacion_fold = df_modelo.loc[
            df_modelo["fecha_emision"] == fecha_origen
        ].copy()


        modelo_fold = clone(pipeline).set_params(**configuracion)

        modelo_fold.fit(
            train_fold[features],
            train_fold[objetivo],
        )

        evaluacion_fold["diferencia_predicha"] = modelo_fold.predict(
            evaluacion_fold[features]
        )

        evaluacion_fold["pronostico_corregido"] = (
            evaluacion_fold["unidades_pronosticadas"]
            - evaluacion_fold["diferencia_predicha"]
        )

        predicciones_folds.append(evaluacion_fold)

    return pd.concat(predicciones_folds, ignore_index=True)


def evaluate_minimum_history(
    df_hist: pd.DataFrame,
    min_dias: int = 45,
) -> pd.DataFrame:
    """
    Count consecutive days per center up to the dataframe's latest date.

    Args:
        df_hist (pd.DataFrame): DataFrame containing historical dispatch data with columns 'centro_distribucion' and 'fecha_despacho'.
        min_dias (int, optional): Minimum number of consecutive days required to enable training. Defaults to 45.
    Returns:
        pd.DataFrame: DataFrame with columns 'centro_distribucion', 'dias_continuos', and 'habilita_entrenamiento' indicating the number of consecutive days and whether training is enabled for each distribution center.
    """

    hist = df_hist.copy()

    hist["fecha_despacho"] = pd.to_datetime(
        hist["fecha_despacho"],
        errors="coerce",
    ).dt.normalize()

    ultima_fecha = hist["fecha_despacho"].max()
    resultados = []

    for centro, datos in hist.groupby("centro_distribucion"):

        fechas_disponibles = set(datos["fecha_despacho"].dropna())

        dia = ultima_fecha
        dias_continuos = 0

        # Count backward until a missing date is found.
        while dia in fechas_disponibles:
            dias_continuos += 1
            dia -= pd.Timedelta(days=1)

        resultados.append({
            "centro_distribucion": centro,
            "dias_continuos": dias_continuos,
            "habilita_entrenamiento": dias_continuos >= min_dias,
        })

    return pd.DataFrame(
        resultados,
        columns=[
            "centro_distribucion",
            "dias_continuos",
            "habilita_entrenamiento",
        ],
    )

def train_model_with_gate(
    df_modelo: pd.DataFrame,
    estado_centros: pd.DataFrame,
    pipeline: Pipeline,
    configuracion: dict,
    features: list[str],
    objetivo: str,
) -> Pipeline | None:
    """Train with enabled centers using closed history.

    Return None if no usable training records are available.

    Args:
        df_modelo (pd.DataFrame): DataFrame containing the model data with features and target
        estado_centros (pd.DataFrame): DataFrame containing the status of distribution centers with columns 'centro_distribucion' and 'habilita_entrenamiento'.
        pipeline (Pipeline): A scikit-learn pipeline containing preprocessing and model steps.
        configuracion (dict): Configuration parameters for the model.
        features (list[str]): List of feature column names to be used for training.
        objetivo (str): Name of the target column to be predicted.
    Returns:
        Pipeline | None: Trained model pipeline if training data is available, otherwise None.
    
    """

    centros_habilitados = estado_centros.loc[
        estado_centros["habilita_entrenamiento"],
        "centro_distribucion",
    ]

    datos_entrenamiento = df_modelo.loc[
        df_modelo["centro_distribucion"].isin(centros_habilitados)
    ].dropna(subset=features + [objetivo])

    if datos_entrenamiento.empty:
        return None

    modelo = clone(pipeline).set_params(**configuracion)

    modelo.fit(
        datos_entrenamiento[features],
        datos_entrenamiento[objetivo],
    )

    return modelo


def generate_final_forecast(
    df_prediccion: pd.DataFrame,
    estado_centros: pd.DataFrame,
    modelo: Pipeline | None,
    features: list[str],
) -> pd.DataFrame:
    """
    Apply the model when possible and keep the baseline as a fallback.
    
    Args:
        df_prediccion (pd.DataFrame): DataFrame containing the prediction data with features and forecasted units.
        estado_centros (pd.DataFrame): DataFrame containing the status of distribution centers with columns 'centro_distribucion' and 'habilita_entrenamiento'.
        modelo (Pipeline | None): Trained model pipeline if training data is available, otherwise None
        features (list[str]): List of feature column names to be used for prediction.
    Returns:
        pd.DataFrame: DataFrame containing the original prediction data along with the final forecasted units and the source of the forecast (baseline or complementary model).
    """

    resultado = df_prediccion.copy().reset_index(drop=True)

    estados = estado_centros.set_index(
        "centro_distribucion"
    )["habilita_entrenamiento"]

    resultado["habilita_entrenamiento"] = (
        resultado["centro_distribucion"]
        .map(estados)
        .eq(True)
    )

    resultado["pronostico_final"] = resultado["unidades_pronosticadas"]
    resultado["fuente_pronostico"] = "base"

    if modelo is None:
        return resultado

    X = resultado.reindex(columns=features)

    puede_predecir = (
        resultado["habilita_entrenamiento"]
        & np.isfinite(X).all(axis=1)
    )

    if not puede_predecir.any():
        return resultado

    diferencia = modelo.predict(X.loc[puede_predecir])

    corregido = (
        resultado.loc[puede_predecir, "unidades_pronosticadas"]
        - diferencia
    )

    corregido = corregido.loc[np.isfinite(corregido)]

    resultado.loc[corregido.index, "pronostico_final"] = (
        corregido.clip(lower=0)
    )
    resultado.loc[corregido.index, "fuente_pronostico"] = (
        "modelo_complementario"
    )

    return resultado
