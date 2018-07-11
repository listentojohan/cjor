import json
import math
import os
import googlemaps
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pprint import pprint

"""Input a centroid and radius. Returns hexagon as polygon"""

KEY = os.getenv('GMAPS_KEY')
gmaps = googlemaps.Client(KEY)
earth_r=6378.1 #Radius of the Earth
brng = math.radians(60)# 1.57 #Bearing is 90 degrees converted to radians.
radius = 15 #Distance in km


def new_lat_lon(lon1,lat1, brng):
    lat2 = math.asin(math.sin(lat1)*math.cos(distance/earth_r) + 
        math.cos(lat1)*math.sin(distance/earth_r)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distance/earth_r)*math.cos(lat1),
        math.cos(distance/earth_r)-math.sin(lat1)*math.sin(lat2))
    lat, lon = math.degrees(lat2), math.degrees(lon2)
    return (lat,lon)

def hex_from_centroid(centroid: Point, radius_in_km: float): 
    global distance 
    distance = radius_in_km
    brng = 0
    poly = []
    for i in range(0,6):
        poly.append(new_lat_lon(lon1=math.radians(centroid.x),
                                lat1=math.radians(centroid.y), 
                                brng=math.radians(brng)))
        brng+=60
    # add start point to close of polygon
    poly.append(poly[0])
    return Polygon(poly)

def lat_lon_from_address(place: str):
    """Returns dict w. lat, lon"""
    resp = gmaps.geocode(address=place)
    latlon_dict = resp[0]['geometry']['location'] 
    return (latlon_dict['lat'], latlon_dict['lng'])

if __name__=="__main__":
    df = pd.read_csv('input.csv') # except column w. city, country and radius
    df['latlng'] = None
    df['geometry'] = None
    for idx, row in df.iterrows():
        #print(idx, row.city) 
        address = '{}, {}'.format(row.city, row.country)
        latlng = lat_lon_from_address(address)
        df.at[idx,'latlng'] = latlng
        df.at[idx,'geometry'] = hex_from_centroid(Point(latlng), row.radius)
        #print(row.polygon)
    #df.to_csv('file.csv')
    df.drop(['latlng'],axis=1, inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=df.geometry)
    #gdf.to_file('geo_french_cities.geojson',driver='GeoJSON')
    _json = json.loads(gdf.to_json())
    for feature in _json['features']:
        del feature['id']
        city = str("City - "+feature['properties']['city'].title())
        feature['properties'].update({"name":city})
        with open('cities/'+feature['properties']['city']+'.json', 'w') as out:
            json.dump(feature, out)

