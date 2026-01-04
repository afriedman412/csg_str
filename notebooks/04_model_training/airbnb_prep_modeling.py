# airbnb / insideairbnb data cleaning and modeling code

import pandas as pd
import re
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

import censusdata

import lightgbm as lgb

"""PATHING AND COLUMNS"""
PATH_ = "/content/drive/MyDrive/capstone/"

RAW_PATH = 'inside_airbnb_feature_eng_102225.csv'
TO_MODEL_PATH = 'inside_airbnb_to_model_102225.csv'
CENSUS_PATH = "census_data_final_102825.csv"

BUILDING_COLS = ['building_bed and breakfast', 'building_boutique hotel',
                 'building_bungalow', 'building_condo', 'building_cottage',
                 'building_guest suite', 'building_guesthouse', 'building_home',
                 'building_hotel', 'building_loft', 'building_other',
                 'building_rental unit', 'building_resort',
                 'building_serviced apartment', 'building_townhouse', 'building_villa']

MODEL_COLS = ['accommodates',
              'air_conditioning',
              # 'amenities',
              # 'amenities_bool',
              'availability_30',
              'availability_365',
              'availability_60',
              'availability_90',
              'availability_eoy',
              'avg_price',
              'bathrooms',
              # 'bathrooms_text',
              'bedrooms',
              'beds',
              'building_bed and breakfast',
              'building_boutique hotel',
              'building_bungalow',
              'building_condo',
              'building_cottage',
              'building_guest suite',
              'building_guesthouse',
              'building_home',
              'building_hotel',
              'building_loft',
              'building_other',
              'building_rental unit',
              'building_resort',
              'building_serviced apartment',
              'building_townhouse',
              'building_villa',
              'calculated_host_listings_count',
              'calculated_host_listings_count_entire_homes',
              'calculated_host_listings_count_private_rooms',
              'calculated_host_listings_count_shared_rooms',
              # 'calendar_last_scraped',
              # 'calendar_updated',
              'commute_time_mean',
              # 'description',
              'education_bachelors',
              'estimated_occupancy_l365d',
              'estimated_revenue_l365d',
              'first_review',
              'free_parking',
              'gini_index',
              'gym',
              # 'has_availability',
              'heating',
              # 'host_about',
              # 'host_acceptance_rate',
              # 'host_has_profile_pic',
              # 'host_id',
              # 'host_identity_verified',
              # 'host_is_superhost',
              'host_listings_count',
              # 'host_location',
              # 'host_name',
              # 'host_neighbourhood',
              # 'host_picture_url',
              # 'host_response_rate',
              # 'host_response_time',
              'host_response_time_top',
              # 'host_since',
              # 'host_thumbnail_url',
              'host_total_listings_count',
              # 'host_url',
              # 'host_verifications',
              'hot_tub',
              'hot_tub_private',
              'hot_tub_shared',
              'housekeeping',
              # 'id',
              # 'index',
              'instant_bookable',
              'labor_force',
              'last_review',
              # 'last_scraped',
              'latitude',
              # 'license',
              # 'listing_url',
              'longitude',
              'maximum_maximum_nights',
              'maximum_minimum_nights',
              'maximum_nights',
              'maximum_nights_avg_ntm',
              'med_price',
              'median_age',
              'median_gross_rent',
              'median_home_value',
              'median_income',
              'median_year_built',
              'minimum_maximum_nights',
              'minimum_minimum_nights',
              'minimum_nights',
              'minimum_nights_avg_ntm',
              'neighborhood_overview',
              'neighbourhood_cleansed_top',
              'number_of_reviews',
              'number_of_reviews_l30d',
              'number_of_reviews_ltm',
              'number_of_reviews_ly',
              'paid_parking',
              'percent_foreign_born',
              'percent_over_65',
              'percent_owner_occupied',
              'percent_public_transport',
              'percent_work_from_home',
              'picture_url',
              'pool',
              'pool_indoor',
              'pool_outdoor',
              'pool_private',
              'pool_shared',
              'population',
              'poverty_rate',
              'price',
              # 'property_type',
              'race_asian',
              'race_black',
              'race_other',
              'race_white',
              'review_scores_accuracy',
              'review_scores_checkin',
              'review_scores_cleanliness',
              'review_scores_communication',
              'review_scores_location',
              'review_scores_rating',
              'review_scores_value',
              'reviews_per_month',
              # 'room_type',
              # 'scrape_id',
              # 'slice',
              # 'source',
              'state',
              'total_housing_units',
              'unemployed',
              'unemployment_rate',
              'vacancy_rate',
              # 'zip'
              ]


# Define columns to drop later
EXCESS_COLS = [
    'tv_amazon prime video', 'tv_apple tv', 'tv_chromecast',
    'tv_disney+', 'tv_dvd player', 'tv_fire tv', 'tv_hbo max',
    'tv_hulu', 'tv_netflix', 'tv_premium cable', 'tv_roku',
    'tv_standard cable', 'coffee_drip coffee maker',
    'coffee_espresso machine', 'coffee_french press',
    'coffee_keurig coffee machine', 'coffee_nespresso',
    'coffee_pour-over coffee', 'clothing_storage_closet',
    'clothing_storage_dresser', 'clothing_storage_walk-in closet',
    'clothing_storage_wardrobe', 'clothing_storage',
    'fireplace_electric', 'fireplace_ethanol', 'fireplace_gas',
    'fireplace_pellet stove', 'fireplace_wood-burning', 'Unnamed: 0',
    'tv_x', 'tv_y', 'TV', 'air conditioning', 'central air conditioning',
    'central heating', 'fast wifi'
]

"""DATA LOADING CODE"""


def load_df(p):
    df = pd.read_csv(os.path.join(PATH_, p))
    return df


def load_top_n(df, col, N):
    topN = df[col].value_counts().nlargest(N).index
    df[col + "_top"] = df[col].where(df[col].isin(topN), "other")
    return df


def load_top_n(df, col, N):
    topN = df[col].value_counts().nlargest(N).index
    df[col + "_top"] = df[col].where(df[col].isin(topN),
                                     "other").astype("category")
    return df


def amenities_parser(a):
    amenities_dict = {}
    if 'parking' in a:
        if 'free' in a:
            amenities_dict['free_parking'] = True
        elif 'paid' in a:
            amenities_dict['paid_parking'] = True
    if 'air conditioning' in a:
        if not a.startswith('no'):
            amenities_dict['air_conditioning'] = True
    if 'heating' in a:
        amenities_dict['heating'] = True
    if re.search(r"\spool", a):
        amenities_dict['pool'] = True
        for i in ['shared', 'private', 'indoor', 'outdoor']:
            if i in a:
                amenities_dict[f'pool_{i}'] = True
    if 'hot tub' in a:
        amenities_dict['hot_tub'] = True
        for i in ['shared', 'private']:
            if i in a:
                amenities_dict[f'hot_tub_{i}'] = True
    if 'gym' in a:
        amenities_dict['gym'] = True
    if re.search(r"wifi – \d+ mbps", a):
        amenities_dict['wifi'] = True
    if 'housekeeping' in a:
        amenities_dict['housekeeping'] = True
    return amenities_dict


def preprep_df():
    raw_df = load_df(RAW_PATH)
    df_model = load_df(TO_MODEL_PATH)

    raw_df = load_top_n(raw_df, 'neighbourhood_cleansed', 40)
    raw_df = load_top_n(raw_df, 'host_response_time', 4)
    state_dummies = pd.get_dummies(
        raw_df['slice'].fillna("-").map(lambda s: s.split("-")[-1]).replace("island", "ri"), prefix="state", dtype="int8"
    )

    raw_df['price'] = raw_df['price'].fillna("0").map(
        lambda p: re.sub(r"\$|,", "", p)).astype(float)

    # Clean 'minimum_nights'
    raw_df['minimum_nights'] = raw_df['minimum_nights'].map(
        lambda n: re.sub(r"\d{4}\-\d{2}\-\d{2}", '999',
                         n) if isinstance(n, str) else n
    ).fillna(999).astype(int)

    # Convert 'review_scores_rating' to numeric
    raw_df['review_scores_rating'] = pd.to_numeric(
        raw_df['review_scores_rating'], errors="coerce")

    city_stats = (
        raw_df.groupby("slice")["price"]
        .agg(avg_price="mean", n="size", med_price="median")
        .reset_index()
        .sort_values("avg_price", ascending=False)
    )
    raw_df = raw_df.merge(city_stats, left_on="slice", right_on="slice").merge(
        state_dummies, left_index=True, right_index=True)
    raw_df[BUILDING_COLS] = model_df[BUILDING_COLS]
    raw_df['amenities_bool'] = raw_df['amenities'].fillna(
        "").str.lower().map(amenities_parser)

    return raw_df


"""CENSUS CODE"""
vars_ = {
    "median_income": "B19013_001E",
    "median_gross_rent": "B25064_001E",
    "population": "B01003_001E",
    "median_home_value": "B25077_001E",
    "education_bachelors": "B15003_022E",
    "median_age": "B01002_001E",
    "race_white": "B02001_002E",
    "race_black": "B02001_003E",
    "race_asian": "B02001_005E",
    "race_other": "B02001_007E",
    "median_year_built": "B25035_001E",
    "total_housing_units": "B25001_001E",
    "labor_force": "B23025_002E",
    "unemployed": "B23025_005E",
    "commute_time_mean": "B08303_001E",
    "gini_index": "B19083_001E"

}

derived = {
    # each entry is either a tuple (num, denom) or list of vars to sum
    "percent_foreign_born": ("B05002_013E", "B05002_001E"),
    "unemployment_rate": ("B23025_005E", "B23025_002E"),
    "percent_owner_occupied": ("B25003_002E", "B25003_001E"),
    "percent_public_transport": ("B08301_010E", "B08301_001E"),
    "percent_work_from_home": ("B08301_021E", "B08301_001E"),
    "percent_over_65": [
        # sum these male + female bins for age ≥ 65
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E"
    ],
    "vacancy_rate": ("B25002_003E", "B25002_001E"),
    "poverty_rate": ("B17001_002E", "B17001_001E"),
}


def derive_acs_features(df):
    out = pd.DataFrame(index=df.index)
    for k in vars_:
        out[k] = df[vars_[k]]

    # ratios
    out["percent_foreign_born"] = df["B05002_013E"] / df["B05002_001E"]
    out["unemployment_rate"] = df["B23025_005E"] / df["B23025_002E"]
    out["percent_owner_occupied"] = df["B25003_002E"] / df["B25003_001E"]
    out["percent_public_transport"] = df["B08301_010E"] / df["B08301_001E"]
    out["percent_work_from_home"] = df["B08301_021E"] / df["B08301_001E"]
    out['vacancy_rate'] = df["B25002_003E"]/df["B25002_001E"]
    out['poverty_rate'] = df["B17001_002E"]/df["B17001_001E"]

    # sum across multiple fields
    age_cols = [
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E",
        "B01001_024E", "B01001_025E", "B01001_044E", "B01001_045E",
        "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E"
    ]
    out["percent_over_65"] = df[age_cols].sum(axis=1) / df["B01003_001E"]

    return out


def get_census_data():
    all_vars = list(vars_.values())
    for val in derived.values():
        if isinstance(val, tuple):
            all_vars.extend(val)
        elif isinstance(val, list):
            all_vars.extend(val)
    all_vars = sorted(set(all_vars))
    acs = censusdata.download(
        'acs5', 2023,
        censusdata.censusgeo([('zip code tabulation area', '*')]),
        all_vars
    )
    acs_clean = derive_acs_features(acs)
    acs_clean.reset_index(inplace=True)
    acs_clean['zip'] = acs_clean['index'].astype(str).str.extract(r'(\d{5})')
    return acs_clean


class LGBMTrainer:
    def __init__(self, df, target_col, log_target=False):
        """
        Initialize trainer with data.

        Args:
            df (pd.DataFrame): The input dataframe.
            target_col (str): Name of target column.
            log_target (bool): Whether to log-transform the target.
        """
        self.df = df
        self.target_col = target_col
        self.log_target = log_target
        self.cities = self.df['slice']
        self.df.drop(columns='slice', inplace=True)

        # Will be filled during training
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = None
        self.rmse_by_city = None
        self.residuals_by_city = None
        self.feature_importance = None

        self._test_frame = None  # cached test frame

    def train(self, train_test_split_params, lgb_params, num_boost_round=1000, early_stopping_rounds=50, log_period=50):
        """
        Train LightGBM model with provided parameters.
        """
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, **train_test_split_params
        )

        # Optionally log-transform target
        if self.log_target:
            y_train_used = np.log1p(self.y_train)
            y_test_used = np.log1p(self.y_test)
        else:
            y_train_used, y_test_used = self.y_train, self.y_test

        train_data = lgb.Dataset(self.X_train, label=y_train_used)
        valid_data = lgb.Dataset(self.X_test, label=y_test_used)

        self.model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=log_period),
            ],
        )

        # Predict
        self.y_pred = self.model.predict(
            self.X_test, num_iteration=self.model.best_iteration)
        if self.log_target:
            self.y_pred = np.expm1(self.y_pred)

        self._evaluate()

    def _evaluate(self):
        """Compute evaluation metrics after training."""
        y_test_aligned = self.y_test.reset_index(drop=True)
        y_pred_aligned = pd.Series(self.y_pred).reset_index(drop=True)
        slice_aligned = self.cities.loc[self.y_test.index].reset_index(
            drop=True)

        df_eval = pd.DataFrame({
            'slice': slice_aligned,
            'y': y_test_aligned,
            'y_pred': y_pred_aligned
        })

        # RMSE per city
        self.rmse_by_city = (
            df_eval.groupby('slice')
            .apply(lambda x: root_mean_squared_error(x['y'], x['y_pred']))
            .reset_index(name='rmse')
        )

        # Residuals per city
        self.residuals_by_city = (
            df_eval['y'] - df_eval['y_pred']).groupby(df_eval['slice']).mean()

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)

    def get_test_frame(self, include_features: bool = False, recalc: bool = False, copy: bool = True) -> pd.DataFrame:
        """
        Return a tidy DataFrame with y_test, y_pred, residuals, and slice (city).
        Optionally include X_test feature columns for custom plotting.

        Args:
            include_features (bool): If True, append the X_test feature columns.
            recalc (bool): If True, force recomputation even if cached.
            copy (bool): If True, return a copy to avoid accidental mutation.

        Returns:
            pd.DataFrame with columns:
                ['y', 'y_pred', 'residual', 'abs_error', 'pct_error', 'slice', *(features if requested)]
        """
        if self.y_pred is None or self.y_test is None or self.X_test is None:
            raise ValueError("Model not trained yet. Run .train() first.")

        # Serve from cache when possible
        if self._test_frame is not None and not recalc:
            df_out = self._test_frame
        else:
            idx = self.y_test.index
            y_true = self.y_test.reset_index(drop=True)
            y_pred = pd.Series(
                self.y_pred, name="y_pred").reset_index(drop=True)
            slice_aligned = self.cities.loc[self.y_test.index].reset_index(
                drop=True)
            X_aligned = self.X_test.reset_index(drop=True)

            df_out = pd.DataFrame({
                "y": y_true,
                "y_pred": y_pred,
                "slice": slice_aligned
            })
            df_out["residual"] = df_out["y"] - df_out["y_pred"]
            df_out["abs_error"] = df_out["residual"].abs()
            # Safe percentage error (avoid divide-by-zero)
            eps = 1e-9
            df_out["pct_error"] = (
                df_out["residual"] / (df_out["y"].abs() + eps)) * 100.0

            # Cache the base version without features; features can be appended ad-hoc below
            self._test_frame = df_out

        if include_features:
            # Append feature columns (avoid re-joining if already present)
            feats_missing = [
                c for c in self.X_test.columns if c not in df_out.columns]
            if feats_missing:
                df_out = pd.concat(
                    [df_out, self.X_test.reset_index(drop=True)[feats_missing]], axis=1)

        return df_out.copy() if copy else df_out

    # ---------- Visualization Helpers ----------

    def plot_feature_importance(self, top_n=20):
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Run .train() first.")
        top_feats = self.feature_importance.head(top_n)
        plt.figure(figsize=(8, 6))
        plt.barh(top_feats['feature'][::-1], top_feats['importance'][::-1])
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.show()

    def plot_residuals_by_city(self):
        if self.residuals_by_city is None:
            raise ValueError("Model not trained yet. Run .train() first.")
        self.residuals_by_city.sort_values().plot(kind='barh', figsize=(8, 6))
        plt.title("Mean Residuals by City")
        plt.xlabel("Mean Residual (y - y_pred)")
        plt.show()

    def plot_pred_vs_true(self, add_identity=True, annotate_metrics=True, use_seaborn=True):
        """
        Scatter/regression plot of y_pred vs y_test.

        Args:
            add_identity (bool): Draw y=x reference line.
            annotate_metrics (bool): Show RMSE and R^2 in the title.
            use_seaborn (bool): Try to use seaborn.regplot; falls back to matplotlib scatter.
        """
        if self.y_pred is None or self.y_test is None:
            raise ValueError("Model not trained yet. Run .train() first.")

        # Align types and indices
        y_true = self.y_test.reset_index(drop=True)
        y_pred = pd.Series(self.y_pred).reset_index(drop=True)

        plt.figure(figsize=(6, 6))

        drew_with_seaborn = False
        if use_seaborn:
            try:
                import seaborn as sns
                sns.regplot(
                    x=y_true,
                    y=y_pred,
                    ci=None,
                    scatter_kws={'alpha': 0.3, 's': 10},
                    line_kws={'linewidth': 1},
                )
                drew_with_seaborn = True
            except Exception:
                drew_with_seaborn = False

        if not drew_with_seaborn:
            # Fallback: plain matplotlib
            plt.scatter(y_true, y_pred, alpha=0.3, s=10)
            # Add a simple least-squares fit line
            try:
                coeffs = np.polyfit(y_true, y_pred, 1)
                x_line = np.linspace(y_true.min(), y_true.max(), 100)
                plt.plot(x_line, coeffs[0] * x_line + coeffs[1], linewidth=1)
            except Exception:
                pass

        if add_identity:
            lims = [min(y_true.min(), y_pred.min()),
                    max(y_true.max(), y_pred.max())]
            plt.plot(lims, lims, linestyle='--', linewidth=1)
            plt.xlim(lims)
            plt.ylim(lims)

        title = "LightGBM: y_pred vs y_test"
        if annotate_metrics:
            from sklearn.metrics import r2_score, mean_squared_error
            rmse = (mean_squared_error(y_true, y_pred)) ** 0.5
            r2 = r2_score(y_true, y_pred)
            title += f"\nRMSE={rmse:.3f}, R²={r2:.3f}"

        plt.xlabel("y_test")
        plt.ylabel("y_pred")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # ---------- Convenience Accessors ----------

    def get_summary(self):
        """Return key results as a dict."""
        return {
            "model": self.model,
            "rmse_by_city": self.rmse_by_city,
            "residuals_by_city": self.residuals_by_city,
            "feature_importance": self.feature_importance,
        }
