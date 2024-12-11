import pandas as pd
import wbdata
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Union, Tuple


def fetch_indicators_by_topic(topic_id: int) -> pd.DataFrame:
    """
    Fetches and organizes indicators for a specific World Bank topic.

    Args:
        topic_id (int): The World Bank topic ID to fetch indicators for

    Returns:
        pd.DataFrame: DataFrame containing indicator information with columns:
            - Series ID: Indicator identifier
            - Series Name: Indicator name
            - Series Description: Detailed description
            - Source Organization: Data provider
            - Source ID: Source identifier
    """
    series = pd.json_normalize(wbdata.get_indicators(topic=topic_id))
    series = series.rename(columns={
        "id": "Series ID",
        "name": "Series Name",
        "sourceNote": "Series Description",
        "sourceOrganization": "Source Organization",
        "source.id": "Source ID"
    })
    return series[['Series ID', 'Series Name', 'Series Description',
                   'Source Organization', 'Source ID']]


def search_indicators(query: str) -> pd.DataFrame:
    """
    Searches for World Bank indicators matching a query string.

    Args:
        query (str): Search term for indicators

    Returns:
        pd.DataFrame: DataFrame of matching indicators with the same structure
        as fetch_indicators_by_topic()
    """
    series = pd.json_normalize(wbdata.get_indicators(query=query))
    series = series.rename(columns={
        "id": "Series ID",
        "name": "Series Name",
        "sourceNote": "Series Description",
        "sourceOrganization": "Source Organization",
        "source.id": "Source ID"
    })
    return series[['Series ID', 'Series Name', 'Series Description',
                   'Source Organization', 'Source ID']]


def analyze_regional_data(data_df: pd.DataFrame,
                          region: str,
                          value_column: str) -> Dict[str, Union[float, pd.Series]]:
    """
    Performs statistical analysis on data for a specific region.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data
        region (str): Region to analyze
        value_column (str): Name of the column containing values

    Returns:
        dict: Dictionary containing statistical measures:
            - mean: Regional mean
            - median: Regional median
            - std: Standard deviation
            - quartiles: 25th, 50th, 75th percentiles
            - outliers: Countries with values outside 1.5 * IQR
    """
    regional_data = data_df[data_df['Region'] == region][value_column].astype(float)

    q1 = regional_data.quantile(0.25)
    q3 = regional_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data_df[
        (data_df['Region'] == region) &
        (data_df[value_column].astype(float).apply(
            lambda x: x < lower_bound or x > upper_bound
        ))
        ]

    return {
        'mean': regional_data.mean(),
        'median': regional_data.median(),
        'std': regional_data.std(),
        'quartiles': regional_data.quantile([0.25, 0.5, 0.75]),
        'outliers': outliers
    }


def create_regional_comparison(world_gdf: gpd.GeoDataFrame,
                               value_column: str,
                               region: str,
                               figsize: Tuple[int, int] = (20, 12)) -> Tuple[plt.Figure, Dict]:
    """
    Creates a comprehensive regional comparison visualization including maps and statistics.

    Args:
        world_gdf (gpd.GeoDataFrame): GeoDataFrame with world data
        value_column (str): Column to analyze
        region (str): Region to focus on
        figsize (tuple): Figure size as (width, height)

    Returns:
        tuple: (Figure object, Dictionary of regional statistics)
    """
    # Filter for region
    regional_gdf = world_gdf[world_gdf['Region'] == region].copy()

    # Calculate statistics
    stats = {
        'mean': regional_gdf[value_column].mean(),
        'median': regional_gdf[value_column].median(),
        'std': regional_gdf[value_column].std(),
        'count': len(regional_gdf),
        'min': regional_gdf[value_column].min(),
        'max': regional_gdf[value_column].max()
    }

    # Create visualization
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(2, 2, height_ratios=[2, 1])

    # Map
    ax1 = fig.add_subplot(grid[0, :])
    regional_gdf.plot(
        column=value_column,
        legend=True,
        legend_kwds={'label': value_column},
        cmap='viridis',
        ax=ax1
    )
    ax1.set_title(f'{value_column} in {region}')
    ax1.axis('off')

    # Distribution
    ax2 = fig.add_subplot(grid[1, 0])
    sns.boxplot(data=regional_gdf, y=value_column, ax=ax2)
    ax2.set_title('Distribution')

    # Statistics
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.axis('off')
    stats_text = '\n'.join([f'{k}: {v:.2f}' for k, v in stats.items()])
    ax3.text(0.1, 0.5, f'Statistics:\n\n{stats_text}',
             fontsize=12, verticalalignment='center')

    plt.tight_layout()
    return fig, stats


def create_lending_income_matrix(data_df: pd.DataFrame,
                                 value_column: str) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Creates a matrix visualization comparing lending types and income levels.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data
        value_column (str): Column to analyze

    Returns:
        tuple: (Figure object, DataFrame with aggregated statistics)
    """
    # Create pivot table
    matrix = pd.pivot_table(
        data_df,
        values=value_column,
        index='Income Level',
        columns='Lending Type',
        aggfunc='mean'
    )

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        ax=ax
    )
    ax.set_title(f'Average {value_column} by Income Level and Lending Type')
    plt.tight_layout()

    return fig, matrix


def plot_time_series_by_group(data_df: pd.DataFrame,
                              date_column: str,
                              value_column: str,
                              group_column: str,
                              figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Creates a time series plot grouped by a categorical variable.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data
        date_column (str): Name of the date column
        value_column (str): Name of the value column
        group_column (str): Name of the grouping column
        figsize (tuple): Figure size as (width, height)

    Returns:
        plt.Figure: Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, group in data_df.groupby(group_column):
        ax.plot(
            group[date_column],
            group[value_column],
            label=name,
            marker='o',
            markersize=4,
            linewidth=2,
            alpha=0.7
        )

    ax.set_title(f'{value_column} Over Time by {group_column}')
    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig