import geopandas as gpd
import numpy as np


def build_features(df: gpd.GeoDataFrame) -> list[gpd.GeoDataFrame]:
    """
    Build all features for the df dataset
    """
    df = ratios_filosofi(df)
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df = extreme_price_filters(df)
    df = surfaces(df)
    return distance_Eiffel_tower(df)


def ratios_filosofi(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build ratios features
    """
    df = df.copy()
    df["tx_pauvrete"] = df["nbre_menages_pauvres"] / df["nbre_menages"]
    df["nv_vie_moyen"] = df["somme_nv_vie"] / df["nbre_menages"]
    df["part_proprietaires"] = df["nbre_menages_prop"] / df["nbre_menages"]
    df["part_logements_sociaux"] = df["nbre_logements_sociaux"] / df["nbre_menages"]
    df["part_logements_avant_1945"] = df["nbre_logements_avant_1945"] / df["nbre_menages"]
    df["surface_moyenne_logements"] = df["surface_totale_logements"] / df["nbre_menages"]
    return df


def extreme_price_filters(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build extreme price filter features
    """
    df = df.copy()
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    lower, higher = df["prix_m2"].quantile([0.01, 0.99])
    df = df.loc[df["prix_m2"].between(lower, higher)]
    prix_m2_by_commune = df.groupby("code_commune")["prix_m2"]
    df = df.loc[
        df["prix_m2"].between(
            prix_m2_by_commune.transform("quantile", 0.01),
            prix_m2_by_commune.transform("quantile", 0.99),
        )
    ]
    return df


def surfaces(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build surface features
    """
    df = df.copy()
    df["log_surface"] = np.log(df["surface_reelle_bati"])
    df["log_surface_terrain"] = np.log1p(df["surface_terrain"].fillna(0))
    df["ratio_carrez"] = df["lot1_surface_carrez"] / df["surface_reelle_bati"]
    return df


def distance_Eiffel_tower(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build distance to the Eiffel Tower features
    """
    df = df.copy()
    tour_eiffel = gpd.GeoSeries.from_xy(
        [2.2945],
        [48.8584],
        crs="EPSG:4326"
        ).to_crs("EPSG:2154")
    distance_eiffel_km = (
        df.geometry.to_crs("EPSG:2154").distance(tour_eiffel.iloc[0]) / 1000
    )
    df["log_distance_eiffel"] = np.log1p(distance_eiffel_km)
    return df


