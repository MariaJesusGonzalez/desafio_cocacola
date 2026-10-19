import pandas as pd
import matplotlib.pyplot as plt

def calculate_wape(real: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate the Weighted Absolute Percentage Error (WAPE).
    Args:
        real (pd.Series): Series containing the actual values.
        forecast (pd.Series): Series containing the forecasted values.
    Returns:
        float: WAPE value.
    """
    denominator = real.abs().sum()

    if denominator == 0:
        return float("nan")

    return (real - forecast).abs().sum() / denominator


def calculate_bias(real: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate the forecast Bias.

    Args:
        real (pd.Series): Series containing the actual values.
        forecast (pd.Series): Series containing the forecasted values.
    Returns:
        float: Bias value.
    """
    denominator = real.abs().sum()

    if denominator == 0:
        return float("nan")

    return (real - forecast).sum() / denominator

def calculate_metrics(
    df: pd.DataFrame,
    real_col: str,
    forecast_col: str,
    group_by: list[str] | None = None
) -> pd.DataFrame:
    """
    Calculate WAPE and Bias by selected groups.

    Args:
        df (pd.DataFrame): DataFrame containing the actual and forecasted values.
        real_col (str): Name of the column with actual values.
        forecast_col (str): Name of the column with forecasted values.
        group_by (list[str] | None): List of columns to group by. If None, calculate metrics for the entire DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing WAPE and Bias for each group or overall.
    """

    if group_by is None:
        return pd.DataFrame({
            "WAPE": [calculate_wape(df[real_col], df[forecast_col])],
            "Bias": [calculate_bias(df[real_col], df[forecast_col])]
        })

    results = []

    for keys, group in df.groupby(group_by):

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_by, keys))

        row["WAPE"] = calculate_wape(group[real_col], group[forecast_col])
        row["Bias"] = calculate_bias(group[real_col], group[forecast_col])

        results.append(row)

    results_df = pd.DataFrame(results)

    return results_df


def plot_wape_cd_horizon(
    metrics_cd_horizon: pd.DataFrame,
    metrics_horizon: pd.DataFrame,
    horizons: list[str],
) -> None:
    """
    Plot WAPE by distribution center and forecast horizon.
    
    Args:
        metrics_cd_horizon (pd.DataFrame): DataFrame containing WAPE by distribution center and horizon.
        metrics_horizon (pd.DataFrame): DataFrame containing global WAPE by horizon.
        horizons (list[str]): List of forecast horizons to plot.
    Returns:
        None
    """

    plt.figure(figsize=(10, 6))

    # WAPE by distribution center
    for cd, group in metrics_cd_horizon.groupby("centro_distribucion"):

        group = group.set_index("horizonte").reindex(horizons)

        plt.plot(
            horizons,
            group["WAPE"],
            marker="o",
            label=cd
        )

    # Global WAPE
    global_metrics = metrics_horizon.set_index("horizonte").reindex(horizons)

    plt.plot(
        horizons,
        global_metrics["WAPE"],
        linestyle="--",
        linewidth=3,
        marker="o",
        label="Global"
    )

  
    plt.ylim(0, max(metrics_cd_horizon["WAPE"].max(), metrics_horizon["WAPE"].max()) * 1.1)

    plt.title("WAPE por horizonte de planificación")
    plt.xlabel("Horizonte de planificación")
    plt.ylabel("WAPE")


    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/wape_by_horizon.png", dpi=300)
    plt.show()


def calculate_out_of_plan_metrics(
    df: pd.DataFrame,
    flag_col: str = "fuera_de_plan",
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate metrics related to out-of-plan purchases.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        flag_col (str): Name of the column indicating out-of-plan purchases (0 or 1).
        group_by (list[str] | None): List of columns to group by. If None, calculate metrics for the entire DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing metrics related to out-of-plan purchases.
    """

  
    if not group_by:
        flag = df[flag_col]

        return pd.DataFrame({
            "compras": [len(flag)],
            "compras_fuera_plan": [flag.sum()],
            "porcentaje_fuera_plan": [flag.mean() * 100],
        })


    return df.groupby(
        group_by,
        as_index=False,
        observed=True,
    ).agg(
        compras=(flag_col, "size"),
        compras_fuera_plan=(flag_col, "sum"),
        porcentaje_fuera_plan=(
            flag_col, lambda s: s.mean() * 100
        ),
    )

def calculate_client_plan_bias(df: pd.DataFrame) -> pd.DataFrame:
    """

    calculate volume bias and date deviations.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
    Returns:
        pd.DataFrame: DataFrame containing client-level metrics related to adherence to the plan.
    """

    df_bias = df.copy()
    # 1. Bias of volume by client
    bias_volumen = calculate_metrics(
        df_bias,
        real_col="unidades_reales",
        forecast_col="unidades_plan",
        group_by=["cliente_id"],
    )[["cliente_id", "Bias"]]

    bias_volumen = bias_volumen.rename(
        columns={"Bias": "bias_volumen"}
    )

    # 2. Bias of dates by client
    df_bias["desviacion_dias"] = (
        df_bias["fecha_compra_real"] - df_bias["fecha_plan"]
    ).dt.days

    df_bias["desviacion_absoluta_dias"] = df_bias["desviacion_dias"].abs()


    bias_fechas = df_bias.groupby("cliente_id", as_index=False).agg(
        n_compras=("desviacion_dias", "size"),
        bias_fecha_dias=("desviacion_dias", "mean"),
        desviacion_absoluta_media_dias=("desviacion_absoluta_dias", "mean"),
    )

    bias_clientes = bias_volumen.merge(
        bias_fechas,
        on="cliente_id",
        validate="one_to_one",
    )

    bias_clientes["bias_volumen_pct"] = (
    bias_clientes["bias_volumen"] * 100
    )

    return bias_clientes

def plot_client_plan_bias(bias_clientes: pd.DataFrame) -> None:
    """
    plot client-level metrics related underestimation or overestimation of the plan.
    Args:
        bias_clientes (pd.DataFrame): DataFrame containing client-level metrics related to adherence to the plan.
    Returns:
        None
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        bias_clientes["bias_volumen_pct"].dropna(),
        bins=15,
        edgecolor="white",
    )
    axes[0].set_title("Bias de volumen por cliente")
    axes[0].set_xlabel("Bias de volumen (%)")

    axes[1].hist(
        bias_clientes["bias_fecha_dias"].dropna(),
        bins=15,
        edgecolor="white",
    )
    axes[1].set_title("Sesgo de fechas por cliente")
    axes[1].set_xlabel("Desviación media de fechas (días)")

    for ax in axes:
        ax.axvline(0, color="black", linestyle="--")
        ax.set_ylabel("Número de clientes")

    plt.tight_layout()
    plt.savefig("outputs/client_plan_bias.png", dpi=300)
    plt.show()


def calculate_client_plan_adherence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate client-level metrics related to adherence to the plan.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
    Returns:
        pd.DataFrame: DataFrame containing client-level metrics related to adherence to the plan.
    """

    resumen_clientes_plan = calculate_out_of_plan_metrics(df,
    group_by=["cliente_id"],
    )

    # Adherence
    resumen_clientes_plan["adhesion_pct"] = (
        100 - resumen_clientes_plan["porcentaje_fuera_plan"]
    )

    # Concentration
    total_desvios = resumen_clientes_plan["compras_fuera_plan"].sum()

    resumen_clientes_plan["participacion_desvios_pct"] = (
        100 * resumen_clientes_plan["compras_fuera_plan"] / total_desvios
        if total_desvios > 0
        else 0
    )

    # Show first the customers with the most off-plan
    resumen_clientes_plan = resumen_clientes_plan.sort_values(
        ["compras_fuera_plan", "cliente_id"],
        ascending=[False, True],
    )
    resumen_clientes_plan = resumen_clientes_plan[[
        "cliente_id",
        "compras",
        "compras_fuera_plan",
        "adhesion_pct",
        "participacion_desvios_pct",
    ]]

    return resumen_clientes_plan
