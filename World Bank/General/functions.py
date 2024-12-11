import pandas as pd
import wbdata

def fetch_and_organize_country_data()->pd.DataFrame:
    """
    Fetches country data using the wbdata package and organizes it into a Pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing details about countries including ID, Name, Region,
                      Income Level, Capital City, Longitude, and Latitude.
    """
    # Step 1: Fetch country data using wbdata
    print("Fetching country data from World Bank using wbdata...")
    countries = wbdata.get_countries()

    # Step 2: Organize the data into a Pandas DataFrame
    print("Organizing data into a DataFrame...")
    list_countries = [
        {
            "ID": country.get("id", ""),
            "Name": country.get("name", ""),
            "Region": country.get("region", {}).get("value", ""),
            "Income Level": country.get("incomeLevel", {}).get("value", ""),
            "Lending Type":country.get("lendingType", {}).get("value", ""),
            "Capital City": country.get("capitalCity", ""),
            "Longitude": country.get("longitude", ""),
            "Latitude": country.get("latitude", "")
        }
        for country in countries
    ]

    df_countries = pd.DataFrame(list_countries)

    return df_countries


#def quick_get_country_info(df_countries:pd.DataFrame,country_name:str):