import geopandas as gpd
import numpy as np


def build_features(df: gpd.GeoDataFrame) -> list[gpd.GeoDataFrame]:
    """
    Build all features for the dvf dataset
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
    dvf = df.copy()
    dvf["tx_pauvrete"] = dvf["nbre_menages_pauvres"] / dvf["nbre_menages"]
    dvf["nv_vie_moyen"] = dvf["somme_nv_vie"] / dvf["nbre_menages"]
    dvf["part_proprietaires"] = dvf["nbre_menages_prop"] / dvf["nbre_menages"]
    dvf["part_logements_sociaux"] = dvf["nbre_logements_sociaux"] / dvf["nbre_menages"]
    dvf["part_logements_avant_1945"] = dvf["nbre_logements_avant_1945"] / dvf["nbre_menages"]
    dvf["surface_moyenne_logements"] = dvf["surface_totale_logements"] / dvf["nbre_menages"]
    return dvf


def extreme_price_filters(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build extreme price filter features
    """
    dvf = df.copy()
    dvf["prix_m2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]
    lower, higher = dvf["prix_m2"].quantile([0.01, 0.99])
    dvf = dvf.loc[dvf["prix_m2"].between(lower, higher)]
    prix_m2_by_commune = dvf.groupby("code_commune")["prix_m2"]
    dvf = dvf.loc[
        dvf["prix_m2"].between(
            prix_m2_by_commune.transform("quantile", 0.01),
            prix_m2_by_commune.transform("quantile", 0.99),
        )
    ]
    return dvf


def surfaces(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build surface features
    """
    dvf = df.copy()
    dvf["log_surface"] = np.log(dvf["surface_reelle_bati"])
    dvf["log_surface_terrain"] = np.log1p(dvf["surface_terrain"].fillna(0))
    dvf["ratio_carrez"] = dvf["lot1_surface_carrez"] / dvf["surface_reelle_bati"]
    return dvf


def distance_Eiffel_tower(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build distance to the Eiffel Tower features
    """
    dvf = df.copy()
    tour_eiffel = gpd.GeoSeries.from_xy(
        [2.2945],
        [48.8584],
        crs="EPSG:4326"
        ).to_crs("EPSG:2154")
    distance_eiffel_km = (
        df.geometry.to_crs("EPSG:2154").distance(tour_eiffel.iloc[0]) / 1000
    )
    df["log_distance_eiffel"] = np.log1p(distance_eiffel_km)
    return dvf


