import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.optimize import curve_fit
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
pd.options.display.max_columns = 50
warnings.filterwarnings("ignore")

df = pd.read_csv("2018.csv", low_memory = False)
#print('Dataframe dimensions:', df.shape)
#tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
#tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
#tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
						 #.T.rename(index={0:'null values (%)'}))
#print(tab_info)
df = df[df.FL_DATE.str.contains('^2018-01')]
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#print(df.shape)
#print(df[0:10][:])
attributes_to_remove = ['OP_CARRIER_FL_NUM', 'TAXI_OUT', 'WHEELS_ON', 'WHEELS_OFF', 'CANCELLED', 
						'CANCELLATION_CODE', 'DIVERTED', 'AIR_TIME', 'DISTANCE', 'CARRIER_DELAY', 
						'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
df.drop(attributes_to_remove, axis = 1, inplace = True)
df = df[['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST',
		'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_IN', 
		'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
		'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME']]
#print(df[:5])
def get_stats(group):
	return {'min': group.min(), 'max': group.max(),
			'count': group.count(), 'mean': group.mean()}
#global_stats = df['TAXI_IN'].groupby(df['OP_CARRIER']).apply(get_stats).unstack()
#global_stats = global_stats.sort_values('count')
#print(global_stats)
def format_hour_minute(time):
    if pd.isnull(time):
        return np.nan
    else:
        if time == 2400: time = 0
        time = "{0:04d}".format(int(time))
        hour_minute = datetime.time(int(time[0:2]), int(time[2:4]))
        return hour_minute
def combine_date_hour_minute(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['FL_DATE', col]].iterrows():    
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0,0)
            liste.append(combine_date_hour_minute(cols))
        else:
            cols[1] = format_hour_minute(cols[1])
            liste.append(combine_date_hour_minute(cols))
    return pd.Series(liste)

df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['FL_DATE'] = create_flight_time(df, 'CRS_DEP_TIME')
df['DEP_TIME'] = df['DEP_TIME'].apply(format_hour_minute)
df['CRS_ARR_TIME'] = df['CRS_ARR_TIME'].apply(format_hour_minute)
df['ARR_TIME'] = df['ARR_TIME'].apply(format_hour_minute)

df_train = df[df['FL_DATE'].apply(lambda x:x.date()) < datetime.date(2018, 1, 23)]
df_test  = df[df['FL_DATE'].apply(lambda x:x.date()) > datetime.date(2018, 1, 23)]
df = df_train