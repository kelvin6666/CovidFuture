import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
from streamlit_folium import folium_static
import folium
import datetime
import calendar
import plotly.express as px
import geopandas as gp

st.set_page_config(layout="wide")


cases_msia = pd.read_csv('dataset/cases_malaysia.csv')
cases_state = pd.read_csv('dataset/cases_state.csv')
clusters = pd.read_csv('dataset/clusters.csv')
tests_msia = pd.read_csv('dataset/tests_malaysia.csv')
tests_state = pd.read_csv('dataset/tests_state.csv')
boruta_rank = pd.read_csv('dataset/boruta_rank.csv')
rfe_rank= pd.read_csv('dataset/rfe_rank.csv')

df_msia = pd.read_csv('dataset/df_msia.csv')
df_map = gp.read_file('msia-states.json')
df_map2 = gp.read_file('msia-states.json')
m = folium.Map([4.602973124617278, 108.64564992244625], zoom_start=5.0)
m2 = folium.Map([4.602973124617278, 108.64564992244625], zoom_start=5.0)



cases_msia.name = 'cases_msia'
cases_state.name = 'cases_state'
clusters.name = 'clusters'
tests_msia.name = 'tests_msia'
tests_state.name = 'tests_state'
csv_list = [cases_msia, tests_msia, cases_state, tests_state, clusters]


for x in csv_list:
  try:
    x['date'] = pd.to_datetime(x['date'],errors='coerce')
  except:
    x['date_announced'] = pd.to_datetime(x['date_announced'],errors='coerce')
    x['date_last_onset'] = pd.to_datetime(x['date_last_onset'],errors='coerce')

cases_msia.fillna(0 ,inplace = True)
cases_state.fillna(0 ,inplace = True)

cases_msia['day'] = cases_msia['date'].dt.day
cases_msia['month'] = cases_msia['date'].dt.month
cases_msia['year'] = cases_msia['date'].dt.year

year2020 = cases_msia['year'] == 2020
cases_msia_2020 = cases_msia[year2020] 
year2021 = cases_msia['year'] == 2021
cases_msia_2021 = cases_msia[year2021]

cases_msia_2020_byMth = cases_msia_2020.groupby(['month'])['cases_new'].sum().reset_index()
cases_msia_2020_byMth['month'] = cases_msia_2020_byMth['month'] .apply(lambda x: calendar.month_name[x])
cases_msia_2021_byMth = cases_msia_2021.groupby(['month'])['cases_new'].sum().reset_index()
cases_msia_2021_byMth['month'] = cases_msia_2021_byMth['month'] .apply(lambda x: calendar.month_name[x])
cases_msia_2021_byMth = cases_msia_2021_byMth.append([{'month' : 'October'},{'month' : 'November'},{'month' : 'December'}], ignore_index = True)
cases_varies = cases_msia_2020_byMth.merge(cases_msia_2021_byMth,on='month')
cases_varies.columns = ['Month','Year 2020 Cases','Year 2021 Cases']
cases_melt = cases_varies.melt(id_vars='Month',value_vars=['Year 2020 Cases','Year 2021 Cases'])

value = ['cluster_import',	'cluster_religious',	'cluster_community',	'cluster_highRisk',	'cluster_education',	'cluster_detentionCentre',	'cluster_workplace']
t = cases_msia_2021.groupby(['month'])['cluster_import',	'cluster_religious',	'cluster_community',	'cluster_highRisk',	'cluster_education',	'cluster_detentionCentre',	'cluster_workplace'].sum().reset_index()
t_cases_melt = t.melt(id_vars='month',value_vars=['cluster_import',	'cluster_religious',	'cluster_community',	'cluster_highRisk',	'cluster_education',	'cluster_detentionCentre',	'cluster_workplace'])
plot2 = px.line(t_cases_melt,x='month',y = 'value',title='Cluster Trend',color='variable')

cases_state['day'] = cases_state['date'].dt.day
cases_state['month'] = cases_state['date'].dt.month
cases_state['year'] = cases_state['date'].dt.year

cases_state_sum = cases_state.groupby(['state'])['cases_new'].sum().reset_index()
state2020 = cases_state['year'] == 2020
cases_state_2020 = cases_state[state2020] 

state2021 = cases_state['year'] == 2021
cases_state_2021 = cases_state[state2021] 


cases_state_2020_byState = cases_state_2020.groupby(['state'])['cases_new'].sum().reset_index()
cases_state_2020_byState['state'][13] = "Kuala Lumpur"
cases_state_2020_byState['state'][14] = "Labuan"
cases_state_2020_byState['state'][15] = "Putrajaya"
cases_state_2020_byState = cases_state_2020_byState.sort_values(by=['state']).reset_index()

cases_state_2021_byState = cases_state_2021.groupby(['state'])['cases_new'].sum().reset_index()
cases_state_2021_byState['state'][13] = "Kuala Lumpur"
cases_state_2021_byState['state'][14] = "Labuan"
cases_state_2021_byState['state'][15] = "Putrajaya"
cases_state_2021_byState = cases_state_2021_byState.sort_values(by=['state']).reset_index()

df_map['Cases'] = cases_state_2020_byState['cases_new']
df_map2['Cases'] = cases_state_2021_byState['cases_new']
bins = list(df_map["Cases"].quantile([0, 0.5, 0.75, 0.95, 1]))
bins2 = list(df_map2["Cases"].quantile([0, 0.5, 0.75, 0.95, 1]))
states = folium.Choropleth(
    geo_data=df_map, 
    data=df_map,
    key_on="feature.properties.name_1",
    columns=['name_1',"Cases"],
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name="Cases",
    bins=bins,
    reset=True,
    ).add_to(m)

states2 = folium.Choropleth(
    geo_data=df_map2, 
    data=df_map2,
    key_on="feature.properties.name_1",
    columns=['name_1',"Cases"],
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name="Cases",
    bins=bins2,
    reset=True,
    ).add_to(m2)


states.geojson.add_child(
    folium.features.GeoJsonTooltip(fields=['name_1', 'Cases'
                                           ],
                                    aliases=['State: ','Cases: '
                                             ])
)

states2.geojson.add_child(
    folium.features.GeoJsonTooltip(fields=['name_1', 'Cases'
                                           ],
                                    aliases=['State: ','Cases: '
                                             ])
)

cases_state_varies = cases_state_2020_byState.merge(cases_state_2021_byState,on='state')
cases_state_varies.drop('index_x',axis='columns',inplace = True)
cases_state_varies.drop('index_y',axis='columns',inplace = True)

cases_state_varies.columns = ['State','Year 2020 Cases','Year 2021 Cases']
cases_state_melt = cases_state_varies.melt(id_vars='State',value_vars=['Year 2020 Cases','Year 2021 Cases'])


tests_msia['day'] = tests_msia['date'].dt.day
tests_msia['month'] = tests_msia['date'].dt.month
tests_msia['year'] = tests_msia['date'].dt.year

tests_msia_2020 = tests_msia[tests_msia['year']==2020]
tests_msia_2021 = tests_msia[tests_msia['year']==2021]

tests_msia_2020 = tests_msia_2020.groupby(['month'])['rtk-ag','pcr'].sum().reset_index()
tests_msia_2021 = tests_msia_2021.groupby(['month'])['rtk-ag','pcr'].sum().reset_index()
tests_msia_2021 = tests_msia_2021.append([{'month' : 10},{'month' : 11},{'month' : 12}], ignore_index = True)

tests_msia_2021['month'] = tests_msia_2021['month'] .apply(lambda x: calendar.month_name[x])
tests_msia_2020['month'] = tests_msia_2020['month'] .apply(lambda x: calendar.month_name[x])
tests_msia_2020.columns = ['month','Year 2020 rtk-ag','Year 2020 pcr']
tests_msia_2021.columns = ['month','Year 2021 rtk-ag','Year 2021 pcr']
tests_msia_2021_melt = tests_msia_2021.melt(id_vars='month',value_vars=['Year 2021 rtk-ag','Year 2021 pcr'])
tests_msia_2020_melt = tests_msia_2020.melt(id_vars='month',value_vars=['Year 2020 rtk-ag','Year 2020 pcr'])
plot3 = px.bar(cases_state_melt,x='State',y = 'value',title='Cases of 2020 by States vs Cases of 2021 by States',color='variable')
plot4 = px.line(tests_msia_2021_melt,x='month',y = 'value',title='Testing Trend of 2021',color='variable')
plot5e = px.line(tests_msia_2020_melt,x='month',y = 'value',title='Testing Trend of 2020',color='variable')

state_list = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
 'Pulau Pinang', 'Perak', 'Perlis', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu',
 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']

testing2020 = cases_msia[cases_msia['year'] == 2020]
testing2021 = cases_msia[cases_msia['year'] == 2021]


testing2020 = testing2020[['date','cases_new']]
testing2021 = testing2021[['date','cases_new']]
malaysia = pd.concat([testing2020,testing2021])

plot2020 = px.line(testing2020,x='date',y ='cases_new',title='Covid-19 Trend of 2020')
plot2021 = px.line(testing2021,x='date',y ='cases_new',title='Covid-19 Trend of 2021')
malaysiaplot = px.line(malaysia,x='date',y ='cases_new',title='Covid-19 Trend of the Whole Pandemic')
statedf = []

statedf = []
johor = cases_state[cases_state['state'] == 'Johor'].reset_index()
pahang = cases_state[cases_state['state'] == 'Pahang'].reset_index()
for x in state_list:
  statedf.append(cases_state[cases_state['state'] == x].reset_index())


state_list = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
 'Pulau Pinang', 'Perak', 'Perlis', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu',
 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']

statedf = []
johor = cases_state[cases_state['state'] == 'Johor'].reset_index()
pahang = cases_state[cases_state['state'] == 'Pahang'].reset_index()
for x in state_list:
  statedf.append(cases_state[cases_state['state'] == x].reset_index())

def corrof(j,state):
    data = []
    for i in statedf:
        print("State :" + i['state'].iloc[0])
        print(state.corrwith(i,axis=0)[j])
        data.append([i['state'].iloc[0],state.corrwith(i,axis=0)[j]])
        print('\n')
    return data

def getCorrOfDf(data):
    df = pd.DataFrame(data, columns = ['State', 'Correlation (Pearson)'])
    df = df.sort_values(by='Correlation (Pearson)',ascending=False).reset_index()
    df = df.drop(0)
    df = df.drop('index',axis='columns')
    return df

corr_new_cases_data = corrof(1,johor)
corr_new_cases = getCorrOfDf(corr_new_cases_data)

corr_new_recovered_data = corrof(3,johor)
corr_new_recovered = getCorrOfDf(corr_new_recovered_data)

corr_new_import_data = corrof(2,johor)
corr_new_import = getCorrOfDf(corr_new_import_data)

corr_new_cases_data_ph = corrof(1,pahang)
corr_new_cases_ph = getCorrOfDf(corr_new_cases_data_ph)

corr_new_recovered_data_ph = corrof(3,pahang)
corr_new_recovered_ph = getCorrOfDf(corr_new_recovered_data_ph)

corr_new_import_data_ph = corrof(2,pahang)
corr_new_import_ph = getCorrOfDf(corr_new_import_data_ph)

statedfTest = []
johorTest = tests_state[tests_state['state'] == 'Johor'].reset_index()
pahangTest = tests_state[tests_state['state'] == 'Pahang'].reset_index()
for x in state_list:
  statedfTest.append(tests_state[tests_state['state'] == x].reset_index())

def corrofTest(j,state):
    data = []
    for i in statedfTest:
        data.append([i['state'].iloc[0],state.corrwith(i,axis=0)[j]])
    return data

def getCorrOfDfTest(data):
    df = pd.DataFrame(data, columns = ['State', 'Correlation (Pearson)'])
    df = df.sort_values(by='Correlation (Pearson)',ascending=False).reset_index()
    df = df.drop(0)
    df = df.drop('index',axis='columns')
    return df

rtk_data_johor = corrofTest(1,johorTest)
rtk_johor = getCorrOfDfTest(rtk_data_johor)

pcr_data_johor = corrofTest(2,johorTest)
pcr_johor = getCorrOfDfTest(pcr_data_johor)

rtk_data_ph = corrofTest(1,pahangTest)
rtk_ph = getCorrOfDfTest(rtk_data_ph)

pcr_data_ph = corrofTest(2,pahangTest)
pcr_ph = getCorrOfDfTest(pcr_data_ph)
def strong_Corr(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where(x[cols] > 0.8, c1,c2)
    return df1

def strong_accPH(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.7700) | (x[cols] == 0.7500), c1,c2)
    df1['F1-score weighted'] = np.where((x['F1-score weighted'] == 0.7200)|(x['F1-score weighted'] == 0.7400), c1,c2)
    return df1

def strong_accKD(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.7900) | (x[cols] == 0.7500), c1,c2)
    df1['F1-score weighted'] = np.where((x['F1-score weighted'] == 0.7900)|(x['F1-score weighted'] == 0.7600), c1,c2)

    return df1

def strong_accJH(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.7800), c1,c2)
    df1['F1-score weighted'] = np.where((x['F1-score weighted'] == 0.7800), c1,c2)

    return df1

def strong_accSLG(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.9000) | (x[cols] == 0.9000), c1,c2)
    df1['F1-score weighted'] = np.where((x['F1-score weighted'] == 0.9000)|(x['F1-score weighted'] == 0.9100), c1,c2)

    return df1

plot5 = px.line(statedf[0],x='date',y = 'cases_new',title='Johor')
plot6 = px.line(statedf[1],x='date',y = 'cases_new',title='Kedah')
plot7 = px.line(statedf[2],x='date',y = 'cases_new',title='Kelantan')
plot8 = px.line(statedf[3],x='date',y = 'cases_new',title='Melaka')
plot9 = px.line(statedf[4],x='date',y = 'cases_new',title='Negeri Sembilan')
plot10 = px.line(statedf[5],x='date',y = 'cases_new',title='Pahang')
plot11 = px.line(statedf[6],x='date',y = 'cases_new',title='Pulau Pinang')
plot12 = px.line(statedf[7],x='date',y = 'cases_new',title='Perak')
plot13 = px.line(statedf[8],x='date',y = 'cases_new',title='Perlis')
plot14 = px.line(statedf[9],x='date',y = 'cases_new',title='Sabah')
plot15 = px.line(statedf[10],x='date',y = 'cases_new',title='Sarawak')
plot16 = px.line(statedf[11],x='date',y = 'cases_new',title='Selangor')
plot17 = px.line(statedf[12],x='date',y = 'cases_new',title='Terengganu')
plot18 = px.line(statedf[13],x='date',y = 'cases_new',title='Kuala Lumpur')
plot19 = px.line(statedf[14],x='date',y = 'cases_new',title='Labuan')
plot20 = px.line(statedf[15],x='date',y = 'cases_new',title='Putrajaya')

state_list = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
 'Pulau Pinang', 'Perak', 'Perlis', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu',
 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']

pt_msia = df_msia.pivot_table(values=['date', 'state', 'cases_new', '1day', '2day', '3day', '4day', '5day',
       '6day', '1week', '2week', '3week', '4week', '1mth', '2mth', '3mth',
       '4mth', '5mth', '6mth', '7mth', '8mth', '9mth', '10mth', '11mth',
       '1year', '1year1Mth', '1year2Mth', '1year3Mth', '1year4Mth',
       '1year5Mth', '1year6Mth', '1year7Mth'],index='date',columns=['state'])

algo = ['Decision Tree','Random Forest','Support Vector','Linear Regression']
dataPahang = {'R2':[0.831,0.900,0.874,0.055],'MAE':[32.351,24.917,24.917,60.546]}    

dfPahang = pd.DataFrame(dataPahang,index=algo)

algo2 = ['Decision Tree','K Nearest Neightbour','Logistic Regession','Naïve Bayes','Random Forest','Support Vector Classification']
dataPahang2 = {'Accuracy':[0.75,0.74,0.70,0.70,0.77,0.72],'F1-score weighted':[0.74,0.69,0.66,0.67,0.72,0.68]}    

dfPahang2 = pd.DataFrame(dataPahang2,index=algo2)

dataKedah2 = {'Accuracy':[0.74,0.75,0.68,0.66,0.79,0.71],'F1-score weighted':[0.74,0.76,0.69,0.67,0.79,0.72]}    
dfKedah2 = pd.DataFrame(dataKedah2,index=algo2)

dataJohor2 = {'Accuracy':[0.77,0.71,0.76,0.78,0.78,0.75],'F1-score weighted':[0.77,0.71,0.76,0.78,0.78,0.76]}    
dfJohor2 = pd.DataFrame(dataJohor2,index=algo2)

dataSel2 = {'Accuracy':[0.83,0.89,0.9,0.88,0.87,0.9],'F1-score weighted':[0.82,0.89,0.9,0.88,0.87,0.91]}    
dfSel2 = pd.DataFrame(dataSel2,index=algo2)

algo3 = ['Decision Tree Regression','Random Forest Regression','Support Vector Regression','Linear Regression']

dataPahang3 = {'R Squared':[0.831,0.9,0.874,0.055],'Mean Absolute Error':[32.351,24.917,24.917,60.546]}    
dfPahang3 = pd.DataFrame(dataPahang3,index=algo3)

dataKedah3 = {'R Squared':[0.932,0.953,0.96,-0.08],'Mean Absolute Error':[43.182,38.726,40.185,230.335]}    
dfKedah3 = pd.DataFrame(dataKedah3,index=algo3)

dataJohor3 = {'R Squared':[0.833,0.944,0.922,-697.273],'Mean Absolute Error':[88.25,58.257,58.257,5603.531]}    
dfJohor3 = pd.DataFrame(dataJohor3,index=algo3)

dataSel3 = {'R Squared':[0.946,0.956,0.941,0.889],'Mean Absolute Error':[192.202,181.822,181.819,306.07]}    
dfSel3 = pd.DataFrame(dataSel3,index=algo3)

def strong_accRegSLG(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.9460) | (x[cols] == 0.9560), c1,c2)
    df1['Mean Absolute Error'] = np.where((x['Mean Absolute Error'] == 192.2020)|(x['Mean Absolute Error'] == 181.822 ), c1,c2)

    return df1

def strong_accRegJH(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.9440) | (x[cols] == 0.9220), c1,c2)
    df1['Mean Absolute Error'] = np.where((x['Mean Absolute Error'] == 58.2570), c1,c2)

    return df1

def strong_accRegKD(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.9530) | (x[cols] == 0.9600), c1,c2)
    df1['Mean Absolute Error'] = np.where((x['Mean Absolute Error'] == 38.7260)|(x['Mean Absolute Error'] == 40.1850 ), c1,c2)

    return df1

def strong_accRegPH(x):
    c1 = 'background-color: black'
    c2 = 'background-color: none' 

    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = np.where((x[cols] == 0.9000) | (x[cols] == 0.8740), c1,c2)
    df1['Mean Absolute Error'] = np.where((x['Mean Absolute Error'] == 24.9170), c1,c2)

    return df1

############################################################################### streamlit ####################################################################################

st.markdown('## Exploratory Data Analysis')
st.subheader('What is the difference of covid trend in 2020 and 2021 ?')
plot1 = px.line(cases_melt,x='Month',y = 'value',title='Cases of 2020 vs Cases of 2021',color='variable')
st.plotly_chart(plot1,use_container_width=True)
with st.expander("See More"):
    com1,com2 = st.columns((1,1))
    com1.plotly_chart(plot2020,use_container_width=True)
    com2.plotly_chart(plot2021,use_container_width=True)
    st.plotly_chart(malaysiaplot,use_container_width=True)
st.write("From the time series plot of cases of 2020 vs cases of 2021, we can identify the waves of covid-19s. It can be seen that the first wave of covid 19 starts from January 2020 until september 2020. The second wave starts in September 2020 until March 2021 and the third wave starts on march 2021.")
st.write("We can also see that cases in 2021 increase more rapidly compared to 2020 as seen from the gradient of the graph. There is a huge drop of cases in september 2021 due to insufficient data. ")


st.subheader('What is the cluster trend in  2021 ?')
c1,c2 = st.columns((2,1))
c1.plotly_chart(plot2,use_container_width=True)
c2.write("In this graph we can see the trends of different clusters. We can see that the workplace cluster is the leading cause for covid cluster. We can also see that the community has the second most cases. We can see religious cluster peaks at may which may be caused by hari raya holidays but it flattens out after that.")
c2.write("")
c2.write("From the graph, we can also see that the number of covid-19 patients infected at workplace escalated rapidly on January 2021. The value then fall off rapidly starting from February until April. This is probably because of the Movement Control Order which restricts the amount of people going out to work. The numbers then escalted on April until July. This rapid increased in number might caused by the reopening of some business sector around April.")

make_map_responsive= """
<style>
[title~="st.iframe"] { width: 100%}
</style>
"""
st.markdown(make_map_responsive, unsafe_allow_html=True)
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Water Color').add_to(m)
folium.TileLayer('cartodbpositron').add_to(m)
folium.TileLayer('cartodbdark_matter').add_to(m)
folium.LayerControl().add_to(m)
folium.TileLayer('Stamen Terrain').add_to(m2)
folium.TileLayer('Stamen Toner').add_to(m2)
folium.TileLayer('Stamen Water Color').add_to(m2)
folium.TileLayer('cartodbpositron').add_to(m2)
folium.TileLayer('cartodbdark_matter').add_to(m2)
folium.LayerControl().add_to(m2)

st.subheader('Covid-19 Cases According to states in Malaysia')
c1e,c2e = st.columns((1,1))

with c1e:
    c1e.write('2020')   
    folium_static(m)
with c2e:
    c2e.write('2021')
    folium_static(m2)

st.markdown('### From the map illustration above we can see the difference in number of cases for each state in the year 2020 and 2021. We can also see the difference of number of cases with color, as the color approach more red-ish hue it indicates higher covid-19 cases.')
st.plotly_chart(plot3,use_container_width=True)
st.write('From this bar plot, we can clearly see the total number of cases by state for both 2020 and 2021.We can see selangor has the highest number of cases followed by kuala lumpur and sabah.For the year 2020, we can see sabah has the highest followed by selangor and kuala lumpur')
# csc5,csc6,csc7,csc8 = st.columns((1,1,1,1))
st.markdown('')
with st.expander('Read Each State Individually'):
    st.caption('Data as of 09 Sept 2021, 11:59pm')
    csc1,csc2,csc3,csc4 = st.columns((1,1,1,1)) #table 
    #col1 #col2 #col3 #col4

    #### Row 1
    csc1.plotly_chart(plot5,use_container_width=True)
    csc2.plotly_chart(plot6,use_container_width=True)
    csc3.plotly_chart(plot7,use_container_width=True)
    csc4.plotly_chart(plot8,use_container_width=True)
    ### Row 2
    csc1.plotly_chart(plot9,use_container_width=True)
    csc2.plotly_chart(plot10,use_container_width=True)
    csc3.plotly_chart(plot11,use_container_width=True)
    csc4.plotly_chart(plot12,use_container_width=True)
    ### Row 3
    csc1.plotly_chart(plot13,use_container_width=True)
    csc2.plotly_chart(plot14,use_container_width=True)
    csc3.plotly_chart(plot15,use_container_width=True)
    csc4.plotly_chart(plot16,use_container_width=True)
    ### Row 4
    csc1.plotly_chart(plot17,use_container_width=True)
    csc2.plotly_chart(plot18,use_container_width=True)
    csc3.plotly_chart(plot19,use_container_width=True)
    csc4.plotly_chart(plot20,use_container_width=True)
          

plotbox = px.box(cases_state, x="state", y="cases_new",color='state')
c13,c23 = st.columns((1,1))

with c13:   
    st.plotly_chart(plot4,use_container_width=True)
with c23:
    st.plotly_chart(plot5e,use_container_width=True)
st.write("""In this we can see the trends of different covid testing method. 
We generally can see pcr is used more often than rtk-ag. We can see that the amount of Covid-19 testing done in 2021 is more than 2020. The highest amount of covid-19 testing using 
rtk-ag is 2.4 million in August. The sudden plummet in September 2021 is due to insufficient data as the dataset we are using is as of 4 September 2021. In 2020, the testing trend
of rtk-ag remain 0 for 3 months, this is probably because of rtk-ag used has not been widespread due to it only being the start of pandemic. On top of that, the market demand for
rtk-ag is not huge before the covid-19 pandemic. Hence, there isn't much stock available in Malaysia.""")
st.markdown("## Outlier Detection")
st.plotly_chart(plotbox,use_container_width=True)
st.write("We found that in all 16 states that there are outliers in all of them. We can see some state show they have more outliers than other states. We didnt remove outliers as outlier may show us potential trends for the data.")
st.markdown("## Correlation Check")
st.write("Since there are a lot of different independant variables that can measure amd exhibit string correlation with Pahang and Johor, we will look into them one by one as follows:")
st.markdown("### Johor ")

with st.expander(label='Expand me'):
    cc1,cc2,cc3 = st.columns((1,1,1))
    cc1.markdown("#### With Respect to New Cases: ")

    cc1.table(corr_new_cases.style.apply(strong_Corr,axis=None))

    cc2.markdown("#### With Respect to Recovered Cases: ")
    cc2.table(corr_new_recovered.style.apply(strong_Corr,axis=None))

    cc3.markdown("#### With Respect to Import Cases: ")
    cc3.table(corr_new_import.style.apply(strong_Corr,axis=None))

    ct1,ct2= st.columns((1,1))
    ct1.markdown("#### With Respect to rtk-ag Testing: ")
    ct1.table(rtk_johor.style.apply(strong_Corr,axis=None))

    ct2.markdown("#### With Respect to pcr Testing: ")
    ct2.table(pcr_johor.style.apply(strong_Corr,axis=None))


st.write("""With the table above we can conclude that Johor exhibits strong positive correlation with Pulau Pinang, Perak, Kedah, 
Terengganu, Pahang, Kelantan, Sabah, Selangor and Sarawak in terms of new cases.
In terms of recoverd cases, import cases, rtk-ag Testing and pcr Testing, we only found weak and moderate 
correlation between Johor and the other states.""")

st.markdown("### Pahang ")
with st.expander(label='Expand me'):
    cp1,cp2,cp3 = st.columns((1,1,1))
    cp1.markdown("#### With Respect to New Cases: ")
    cp1.table(corr_new_cases_ph.style.apply(strong_Corr,axis=None))

    cp2.markdown("#### With Respect to Recovered Cases: ")
    cp2.table(corr_new_recovered_ph.style.apply(strong_Corr,axis=None))

    cp3.markdown("#### With Respect to Import Cases: ")
    cp3.table(corr_new_import_ph.style.apply(strong_Corr,axis=None))

    ctp1,ctp2= st.columns((1,1))
    ctp1.markdown("#### With Respect to rtk-ag Testing: ")
    ctp1.table(rtk_ph.style.apply(strong_Corr,axis=None))

    ctp2.markdown("#### With Respect to pcr Testing: ")
    ctp2.table(pcr_ph.style.apply(strong_Corr,axis=None))

st.write(""" On the other hand, Pahang exhibits
strong positive correlation with Kedah, Terrengganu, Selangor, Melaka, Perak, Pulau Pinang,
Kelantan, Johor, Kuala lumpur and Sabah in terms of new cases. Pahang also has strong
positive correlation in terms of recovered cases with Kedah, Selangor, Kuala Lumpur, Terengganu
and Putrajaya. With respect to rtk-ag testing, Pahang has strong positive correlation with sabah.
Pahang does not have any strong correlation with the other states in terms of import cases and
pcr testing but has weak and moderate correlation  """)

st.markdown("## Feature Selection")
st.write(""" In this section we focus on extracting meaningful independant variables from the dataset provided by Ministry Of Health
 by using two algorithm which is BORUTA and RFE. Since we are using two algorithm to do feature selection, we will use the independant variables that
 has rank below 30 from each algorithm.""")
st.write(""" From the cases of states dataset, we created a new dataframe which contains history of covid-19 data. 
 For example,we have history of covid-19 data from 1 day before until 1 year 11 month before today's date. The example of the dataframe is shown below:""")
st.dataframe(pt_msia.head())
st.markdown("### Boruta ")
st.write("""We will show the strong features to daily cases extracted by Botura algorithm for Pahang, Kedah, Johor and Selangor by using the dataframe above.""")


with st.expander(label='Strong Features indicated by Boruta'):
    cbc1,cbc2,cbc3,cbc4 = st.columns((1,1,1,1))


    cbc1.write("Pahang")
    cbc1.dataframe(boruta_rank[['FeaturePahang','RankingPahang']].sort_values(by="RankingPahang").reset_index().drop('index',axis='columns'))
    cbc2.write("Kedah")
    cbc2.dataframe(boruta_rank[['FeatureKedah','RankingKedah']].sort_values(by="RankingKedah").dropna().reset_index().drop('index',axis='columns'))
    cbc3.write("Johor")
    cbc3.dataframe(boruta_rank[['FeatureJohor','RankingJohor']].sort_values(by="RankingJohor").dropna().reset_index().drop('index',axis='columns'))
    cbc4.write("Selangor")
    cbc4.dataframe(boruta_rank[['FeatureSelangor','RankingSelangor']].sort_values(by="RankingSelangor").dropna().reset_index().drop('index',axis='columns'))



st.markdown("### RFE ")
st.write("""We will show the strong features to daily cases extracted by RFE algorithm for Pahang, Kedah, Johor and Selangor by using the dataframe above.""")
with st.expander(label='Strong Features indicated by RFE'):
    crc1,crc2,crc3,crc4 = st.columns((1,1,1,1))
    crc1.write("Pahang")
    crc1.dataframe(rfe_rank[['FeaturePahang','RankingPahang']].sort_values(by="RankingPahang").dropna().reset_index().drop('index',axis='columns'))
    crc2.write("Kedah")
    crc2.dataframe(rfe_rank[['FeatureKedah','RankingKedah']].sort_values(by="RankingKedah").dropna().reset_index().drop('index',axis='columns'))
    crc3.write("Johor")
    crc3.dataframe(rfe_rank[['FeatureJohor','RankingJohor']].sort_values(by="RankingJohor").dropna().reset_index().drop('index',axis='columns'))
    crc4.write("Selangor")
    crc4.dataframe(rfe_rank[['FeatureSelangor','RankingSelangor']].sort_values(by="RankingSelangor").dropna().reset_index().drop('index',axis='columns'))

st.markdown("### We respect the working principle of both algorithms hence we would conclude that all of the features above are the strong indicator to predict dailycases in Pahang,Kedah, Johor and Selangor. Since we are taking the results of both algorithm, we would then merge the strong features indicated by both algorithms and drop the duplicated features into a dataframe ")

st.markdown("## Classification V/S Regression")

st.markdown("### Classification ")
from PIL import Image
cmpahang1 = Image.open('dataset/image/classification/dt/pahang.png')
cmpahang2 = Image.open('dataset/image/classification/knn/pahang.png')
cmpahang3 = Image.open('dataset/image/classification/lr/pahang.png')
cmpahang4 = Image.open('dataset/image/classification/nb/pahang.png')
cmpahang5 = Image.open('dataset/image/classification/rf/pahang.png')
cmpahang6 = Image.open('dataset/image/classification/svc/pahang.png')
cmkedah1 = Image.open('dataset/image/classification/dt/kedah.png')
cmkedah2 = Image.open('dataset/image/classification/knn/kedah.png')
cmkedah3 = Image.open('dataset/image/classification/lr/kedah.png')
cmkedah4 = Image.open('dataset/image/classification/nb/kedah.png')
cmkedah5 = Image.open('dataset/image/classification/rf/kedah.png')
cmkedah6 = Image.open('dataset/image/classification/svc/kedah.png')
cmjohor1 = Image.open('dataset/image/classification/dt/johor.png')
cmjohor2 = Image.open('dataset/image/classification/knn/johor.png')
cmjohor3 = Image.open('dataset/image/classification/lr/johor.png')
cmjohor4 = Image.open('dataset/image/classification/nb/johor.png')
cmjohor5 = Image.open('dataset/image/classification/rf/johor.png')
cmjohor6 = Image.open('dataset/image/classification/svc/johor.png')
cmselangor1 = Image.open('dataset/image/classification/dt/selangor.png')
cmselangor2 = Image.open('dataset/image/classification/knn/selangor.png')
cmselangor3 = Image.open('dataset/image/classification/lr/selangor.png')
cmselangor4 = Image.open('dataset/image/classification/nb/selangor.png')
cmselangor5 = Image.open('dataset/image/classification/rf/selangor.png')
cmselangor6 = Image.open('dataset/image/classification/svc/selangor.png')

st.write("Pahang")
with st.expander(label='See Confusion Matrix of Each Model for Pahang'):
    ccm1,ccm2,ccm3 = st.columns((1,1,1))
    ccm1.write("Decision Tree")
    ccm1.image(cmpahang1,channels='RGB',use_column_width=True,)
    ccm2.write("KNN")
    ccm2.image(cmpahang2,channels='RGB',use_column_width=True,)
    ccm3.write("Logistic Regression")
    ccm3.image(cmpahang3,channels='RGB',use_column_width=True,)
    ccm1.write("Naive Bayes")
    ccm1.image(cmpahang4,channels='RGB',use_column_width=True,)
    ccm2.write("Random Forest")
    ccm2.image(cmpahang5,channels='RGB',use_column_width=True,)
    ccm3.write("Support Vector")
    ccm3.image(cmpahang6,channels='RGB',use_column_width=True,)

tbpa1,tbpa2 = st.columns((1,1))
tbpa1.table(dfPahang2.style.apply(strong_accPH,axis=None))
tbpa2.write("Classification model to predict Pahang daily cases. Decision Tree and Random Forest performed the best out of all classification model with an accuracy of 0.75 and 0.77 and f1 score of 0.76 and 0.72 respectively. Eventhough Decision Tree and Random Forest were the best model for these case, the accuracy and f1 score of other model were not far off.")
st.write("Kedah")
with st.expander(label='See Confusion Matrix of Each Model for Kedah'):
    ckm1,ckm2,ckm3 = st.columns((1,1,1))
    ckm1.write("Decision Tree")
    ckm1.image(cmkedah1,channels='RGB',use_column_width=True,)
    ckm2.write("KNN")
    ckm2.image(cmkedah2,channels='RGB',use_column_width=True,)
    ckm3.write("Logistic Regression")
    ckm3.image(cmkedah3,channels='RGB',use_column_width=True,)
    ckm1.write("Naive Bayes")
    ckm1.image(cmkedah4,channels='RGB',use_column_width=True,)
    ckm2.write("Random Forest")
    ckm2.image(cmkedah5,channels='RGB',use_column_width=True,)
    ckm3.write("Support Vector")
    ckm3.image(cmkedah6,channels='RGB',use_column_width=True,)

tbke1,tbke2 = st.columns((1,1))
tbke1.table(dfKedah2.style.apply(strong_accKD,axis=None))
tbke2.write('Classification model to predict Kedah daily cases. K nearest neighbor and random forest performed the best out of all classification model with an accuracy of 0.75 and 0.77 and f1 score of 0.76 and 0.79 respectively. Eventhough KNN and random forest were the best model for these case, the accuracy and f1 score of other model were not far off.')
st.write("Johor")
with st.expander(label='See Confusion Matrix of Each Model for Kedah'):
    cjm1,cjm2,cjm3 = st.columns((1,1,1))
    cjm1.write("Decision Tree")
    cjm1.image(cmjohor1,channels='RGB',use_column_width=True,)
    cjm2.write("KNN")
    cjm2.image(cmjohor2,channels='RGB',use_column_width=True,)
    cjm3.write("Logistic Regression")
    cjm3.image(cmjohor3,channels='RGB',use_column_width=True,)
    cjm1.write("Naive Bayes")
    cjm1.image(cmjohor4,channels='RGB',use_column_width=True,)
    cjm2.write("Random Forest")
    cjm2.image(cmjohor5,channels='RGB',use_column_width=True,)
    cjm3.write("Support Vector")
    cjm3.image(cmjohor6,channels='RGB',use_column_width=True,)

tbj1,tbj2 = st.columns((1,1))
tbj1.table(dfJohor2.style.apply(strong_accJH,axis=None))
tbj2.write('Classification model to predict Johor daily cases. Naive Bayes and random forest performed the best out of all classification model with an accuracy of 0.78 for both and f1 score of 0.78 for both. Eventhough Naive Bayes and random forest were the best model for these case, the accuracy and f1 score of other model were not far off.')

st.write("Selangor")
with st.expander(label='See Confusion Matrix of Each Model for Selangor'):
    csm1,csm2,csm3 = st.columns((1,1,1))
    csm1.write("Decision Tree")
    csm1.image(cmselangor1,channels='RGB',use_column_width=True,)
    csm2.write("KNN")
    csm2.image(cmselangor2,channels='RGB',use_column_width=True,)
    csm3.write("Logistic Regression")
    csm3.image(cmselangor3,channels='RGB',use_column_width=True,)
    csm1.write("Naive Bayes")
    csm1.image(cmselangor4,channels='RGB',use_column_width=True,)
    csm2.write("Random Forest")
    csm2.image(cmselangor5,channels='RGB',use_column_width=True,)
    csm3.write("Support Vector")
    csm3.image(cmselangor6,channels='RGB',use_column_width=True,)

tbs1,tbs2 = st.columns((1,1))
tbs1.table(dfSel2.style.apply(strong_accSLG,axis=None))
tbs2.write('Classification model to predict Selangor daily cases. Logistic regression and support vector classification performed the best out of all classification model with an accuracy of 0.9 for both and f1 score of 0.9 and 0.91 respectively. Eventhough Logistic regression and support vector classification were the best model for these case, the accuracy and f1 score of other model were not far off.')

st.markdown('### Conclusion in the cases we tested in Pahang, Kedah, Johor and selangor. We found that 3 out of 4 cases random forest classification was chosen as the top performing model. We also noticed that the performance of the top model didnt not differ far from other models used. ')

st.write('')
st.markdown("### Regression")

cr1,cr2 = st.columns((1,1))
cr1.write("Pahang")
cr1.table(dfPahang3.style.apply(strong_accRegPH,axis=None))
cr1.write(""" Regression model to predict Pahang daily cases. Support vector regression and random forest regression performed the best out of all regression model used as they scored an R square score of 0.9 and 0.874 respectively and mean absolute error of 24.917 for both models. Linear regression performed terribly in this case as it only achieve a R sqare of 0.055 and mean absolute error of 60.546 so we decide to not use linear regression to predict pahang daily cases. Decision tree regression has similar performance to 
the chosen the chosen models but we are also choosing top 2 best performing models so decision tree was not chosen""")
cr2.write("Kedah")
cr2.table(dfKedah3.style.apply(strong_accRegKD,axis=None))
cr2.write('''Regression model to predict Kedah daily cases. Support vector regression and random forest regression performed the best out of all regression model used as they scored an R square score of 0.953 and 0.96  and mean absolute error of 38.726 and 40.185 respectively. Linear regression performed terribly in this case as it only achieve a R sqare of -0.08 and mean absolute error of 40.185 so we decide to not use linear regression to predict Kedah daily cases. Decision tree regression has similar 
performance to  the chosen the chosen models but we are also choosing top 2 best performing models so decision tree was not chosen''')
cr1.write("Johor")
cr1.table(dfJohor3.style.apply(strong_accRegJH,axis=None))
cr1.write("""Regression model to predict Johor daily cases. Support vector regression and random forest regression performed the best out of all regression model used as they scored an R square score of 0.944 and 0.922 respectively and mean absolute error of 58.257 for both. Linear regression performed terribly in this case as it only achieve a R sqare of -697.273 and mean absolute error of 5603.531 so we decide to not use linear 
regression to predict Johor daily cases. Decision tree did not perform as well as the chosen model in this case so it was not chosen.""")
cr2.write("Selangor")
cr2.table(dfSel3.style.apply(strong_accRegSLG,axis=None))
cr2.write("""Regression model to predict Selangor daily cases. Decision regression and random forest regression performed the best out of all regression model used as they scored an R square score of 0.946 and 0.956  and mean absolute error of 192.202 and 181.822 respectively. Linear regression had an ok performance comparing to other cases but in this case it still didnt out perform other model so it was rejected. Support vector regression has similar 
performance to  the chosen the chosen models but we are also choosing top 2 best performing models so Support vector regression was not chosen""")

st.markdown('### Conclusion in the cases we tested in Pahang, Kedah, Johor and Selangor. We found that 4 out of 4 cases random forest regression was the top performing model and 3 out of 4 cases support vector regression was the top performing model. We found that linear regression had extremely terrible performance for 3 out of 4 cases and only in 1 case it perform mildly good. ')

st.markdown("## Grand Conclusion")
st.markdown("### Which is better classification or regression?")
st.write("""Well we cant compare classification and regression as both outpt are different.
If there is a case where we need to predict a detail predicting of how many number of covid cases we will have tomorrow then it is better to use regression as a regression model is able to give you real values.
If there is a case where we only want to know if tomorrow got low, medium, high or super high covid cases. Then classification is better at doing that.
We may not be able to compare classfication and regression model but we can compare the same models
For classification, we found that Random forest perform the best and for regression case we found that random forest regression perform the best.""")

