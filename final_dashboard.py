import pandas as pd
import numpy as np
import streamlit as st
import hvplot.pandas
import matplotlib.pyplot as plt
import holoviews as hv
import seaborn as sns
from  matplotlib.ticker import PercentFormatter
from scipy.stats import iqr
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from scipy.special import hyp2f1
from bokeh.models import HoverTool
from wpca import WPCA
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from st_aggrid import GridOptionsBuilder, AgGrid


st.set_page_config(layout="wide")

st.title('World Developmental Index and Trends')

@st.cache
def load_data():
  return pd.read_csv("data.csv",index_col = 0)

data = load_data()

#reorder columns
data = data[['dev_index','clusters','ISO-code','Country','GDPc','Population growth','mort','Fertility','Life expectancy','Suicide rate', 'Meat consumption','Urbanization rate','Sex-ratio','population']]


### Scatter Plot

class Mean_pop_one_over_x:
  def __init__(self, lrate = 0.05, niter = 3):
    self.lrate = lrate
    self.niter = niter

  def fit(self,X,y,pop):
    self.a_1 = 0
    self.xmax_ = X.max()
    self.xmin_ = X.min()
    self.ymax_ = y.max()
    self.ymin_ = y.min()
    self.popmax_ = pop.max()

    for i in range(self.niter):
      for x, target, weight in zip(X,y,pop):
        closest_x = self.rootfinder(x,target)
        if target > self.a_1/x:
          self.a_1 += self.lrate*(self.ymax_+self.ymin_)/2*closest_x*weight/self.popmax_
        else:
          self.a_1 -= self.lrate*(self.ymax_+self.ymin_)/2*closest_x*weight/self.popmax_
    self.scale_index(X,y)
    return self

  def predict(self,x):
    return self.a_1/x

  def net_input(self,x):
    return self.a_1/x

  def scale_index(self,X,y):
    median_list = []
    for x, target in zip(X,y):
      closest_x = self.rootfinder(x,target)
      median_list.append(closest_x)
      self.index_one_ = self.arclength(np.max(median_list)) - self.arclength(np.min(median_list))
      self.index_zero_ = self.arclength(np.min(median_list))
    return self

  # you need to scale the data to get the correct residues
  def get_index_res(self,x,y):
    closest_x = self.rootfinder(x,y)
    closest_y = self.a_1/closest_x
    distance_x = (x - closest_x)
    distance_y = (y - closest_y)
    res = np.sqrt(distance_x**2+distance_y**2)
    index = (self.arclength(closest_x) - self.index_zero_)/self.index_one_
    return index, res

  def arclength(self,g):
    gg = g/np.sqrt(self.a_1)
    return -np.sqrt(self.a_1)/gg*(1+gg**4)**1.5*hyp2f1(1,5/4,3/4,-gg**4)

  def rootfinder(self,x,y):
    coeff = [1,-x,0,self.a_1*y,-self.a_1**2]
    roots = np.roots(coeff)
    delta = np.inf
    for root in np.roots(coeff):
      if (root.imag<0.001) & (np.abs(x - root) < delta):
        delta = np.abs(x - root)
        best = root
    return np.real(best)

mean_pop_one_over_x = Mean_pop_one_over_x()
mean_pop_one_over_x.fit(data['GDPc'],data['mort'],data['population'])
xx1 = np.linspace(data['GDPc'].min(),data['GDPc'].max(),100)
xx2 = mean_pop_one_over_x.a_1 / xx1
mean_pop_line = [(a,b) for a,b in zip(xx1,xx2)]

hover = HoverTool(tooltips=[("Country", "@Country"),
                            ("Population", "@population{,}"),
                            ("Cluster", "@clusters"),
                            ("GDP per capita","@GDPc{,}"),
                            ("Infant Mortality","@mort"),
                           ])

co2_vs_gdp_scatterplot = data.assign(pop = lambda which: which.population/600000, axis = 1).hvplot(x='GDPc',
                                                                xlabel = 'GDP per capita',
                                                                xformatter='%.0f',
                                                                y='mort',
                                                                ylabel = 'neonatal infant mortality (per 1000 live births)',
                                                                by='clusters',
                                                                hover_cols=['Country',"population"],
                                                                kind="scatter",
                                                                size='pop',
                                                                legend='top_right',
                                                                alpha=0.7,
                                                                height=500,
                                                                width=500,
                                                                tools=[hover],
                                                                title="Developmental Index measured by the arclength of the blue line"
                                                                ) * hv.Curve(mean_pop_line ).redim(y=hv.Dimension('y', range=(0, data['mort'].max()+1)))\
                                                                                            #.redim(x=hv.Dimension('x', range=(0, data['GDPc'].max())))


col1, padding, col2 = st.columns((20,2,20))
with col1:
    st.markdown("### Combining economic and health data to define clusters and an index.")
    st.write(hv.render(co2_vs_gdp_scatterplot, backend='bokeh'))
    st.markdown("###### Source: data.worldbank.org")

### Bar plot

pipe2 = make_pipeline(
    MinMaxScaler(),
    KNNImputer()
)

X2 = data.drop(columns = ['Country','ISO-code','mort','clusters','dev_index','population'])\
         .rename(columns={'GDPc':'GDP per capita (PPP)'}).copy()

columns_titles = ['Population growth','Fertility', 'GDP per capita (PPP)', 'Meat consumption',
        'Life expectancy', 'Urbanization rate',
        'Suicide rate', 'Sex-ratio']
X2 = X2.reindex(columns=columns_titles)

abc = pd.DataFrame(pipe2.fit_transform(X2),columns = X2.columns)\
          .multiply(data.reset_index()['population'], axis = 'index')\
          .join(data.reset_index()['clusters'])\
          .groupby('clusters').sum()\
          .divide(data.groupby('clusters').population.sum(), axis = 'index')

weights = pd.DataFrame(np.ones_like(abc))
pop_clusters = data.groupby('clusters').population.sum()
weights = weights.multiply(pop_clusters,axis=0)/pop_clusters.mean()

wpca = WPCA()

Y = wpca.fit_transform(abc,weights)

transformed = pd.DataFrame(Y)

av_dev = transformed.apply(lambda x: x**2).sum(axis='columns').mean()

to_plot_pop_clusters = pd.DataFrame(
          transformed.iloc[:,1:].apply(lambda x: x**2).sum(axis='columns')/av_dev,\
          columns=['pca_res']
          )\
          .join(pop_clusters.reset_index())

def plot_inside(col):

  barplots = plt.figure()
  outside = barplots.add_axes((0.1,0.1,0.9,0.8),label='outside')
  inside = barplots.add_axes((0.2,0.32,0.3,0.3),label='inside')

  #scale by the IQR scale
  iqrs = iqr(abc,axis=0)

  # getting the errors

  country_weights = pd.DataFrame(np.ones_like(X2))
  country_weights = country_weights.multiply(data.population,axis=0)/data.population.median()

  country_abc = pd.DataFrame(pipe2.transform(X2),columns = X2.columns)

  country_Y = wpca.transform(country_abc,country_weights)

  first_ = pd.DataFrame(np.zeros_like(country_Y))
  first_[0] = pd.DataFrame(country_Y)[0]
  country_predicted = wpca.inverse_transform(first_)

  country_iqrs = iqr(abc,axis=0)

  error = pd.DataFrame((country_abc-country_predicted)/country_iqrs,columns = X2.columns)\
              .apply(abs)\
              .multiply(data.reset_index()['population'], axis = 'index')\
              .sum(axis = 'index')\
              .divide(data.population.sum())

  sns.barplot(x=wpca.components_[0]/iqrs/2, y = X2.columns, ax = outside, **{'xerr':error/2})
  #barplots.suptitle('This relationship describes most of the variation between countries (error bars are population-weighted by country)',x=0.7)
  outside.xaxis.set_major_formatter(PercentFormatter(1))
  outside.set_xlabel("change on an IQR scale")
  outside.yaxis.tick_right()

  pipe_ = make_pipeline(
      KNNImputer()
  )

  pd.DataFrame(pipe_.fit_transform(X2),columns = X2.columns)\
            .multiply(data.reset_index()['population'], axis = 'index')\
            .join(data.reset_index()['clusters'])\
            .groupby('clusters').sum()\
            .divide(data.groupby('clusters').population.sum(), axis = 'index')\
            [col]\
            .plot(kind = 'line',label=col, ax=inside, title = col)

  inside.set_xticks(range(0, 5))

  return barplots

with col2:
    st.markdown("### This linear relationship describes most of the variation between countries:")
    st.markdown("##### (Countries' importance are weighted by their population. Error bars are individual country variances.)")
    option = st.selectbox(
     'Inside plot:',
     columns_titles,index=7)
    st.write(plot_inside(option))
    st.markdown("##### Sex-ratio is non-monotonic and noisy (e.g., United Arab Emirates has a sex-ratio of 2.56 to 1 and Qatar has 3.39 to 1).")
    st.markdown("###### Source: data.worldbank.org")
    st.markdown("###### Source for Meat consumption, Urbanization rate and Sex-ratio: kaggle.com/datasets/daniboy370/world-data-by-country-2020")


### Map

map_data = data.rename(columns={'GDPc':'GDP per capita (PPP)','population':'Population','mort':'Infant Mortality','dev_index':'Developmental Index','clusters':'Clusters'}).copy()

plasma = [
[0.0,"#0d0887"],
[0.11,"#46039f"],
[0.22,"#7201a8"],
[0.33,"#9c179e"],
[0.44,"#bd3786"],
[0.56,"#d8576b"],
[0.67,"#ed7953"],
[0.78,"#fb9f3a"],
[0.89,"#fdca26"],
[1.0,"#f0f921"]
]

map = go.Figure()
for col in map_data.select_dtypes(include=[np.number]).columns:
  figpx = px.choropleth(map_data.assign(Plot=col),
                      locations = 'ISO-code',
                      color=col,
                      custom_data = map_data.assign(Plot=col),
                      hover_name="Country",
                      hover_data=np.append(['Clusters','Developmental Index','Population'],col),
                      ).update_traces(visible=False)

  map.add_traces(figpx.data)

map.update_layout(
    updatemenus=[
        dict(
          buttons =
              [
                  {
                      "label": k,
                      "method": "update",
                      "args":
                      [
                          {"visible": [t.customdata[0][-1] == k for t in map.data]},
                          {"coloraxis": dict(cmin=map_data[k].quantile(0.04),cmax=map_data[k].quantile(0.96),colorbar=dict(x=1.0),colorscale=plasma)},
                      ],
                  }
                  for k in map_data.select_dtypes(include=[np.number]).columns
              ],
          pad={"r": 10, "t": 10},
          showactive=False,
          font = dict(color='#000000'),
          xanchor="left",
          yanchor="top",
          y=1.03,
          bgcolor = '#478F48',
          bordercolor = '#FFFFFF',
        )
    ]
).update_traces(visible=True,
                selector=lambda t: t.customdata[0][-1]=='Developmental Index'
).update_layout(coloraxis=dict(cmin=map_data['Developmental Index'].quantile(0.04),cmax=map_data['Developmental Index'].quantile(0.96))
).update_layout(
    margin=dict(l=2, r=2, t=15, b=15),
    paper_bgcolor="LightSteelBlue",
).update_layout(coloraxis_colorbar_x=1)
st.markdown("<h3 style='text-align: center;'>Map display of developmental index, clusters and features</h3>", unsafe_allow_html=True)

st.plotly_chart(map,use_container_width=True)

## Table

st.markdown("<h3 style='text-align: center;'>Table lookup</h3>", unsafe_allow_html=True)

map_table = data.assign(**{
                    "dev_index": lambda which: round(which.dev_index,3),
                    "GDPc": lambda which: round(which.GDPc,3),
                    'Population growth': lambda which: round(which['Population growth'],3),
                    'Life expectancy': lambda which: round(which['Life expectancy'],3),
                    })\
                 .drop(columns=['ISO-code'])\
                 .rename(columns={'GDPc':'GDP per capita (PPP)','population':'Population','mort':'Infant Mortality','clusters':'Clusters','dev_index':'Dev_Index'})\
                 .sort_values(by='Dev_Index',ascending=False)\
                 [['Country', 'Dev_Index', 'GDP per capita (PPP)','Population growth','Infant Mortality','Fertility','Life expectancy','Suicide rate', 'Meat consumption','Urbanization rate','Sex-ratio','Population']]

gb = GridOptionsBuilder.from_dataframe(map_table)
gb.configure_default_column(width=100)
gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
gb.configure_side_bar() #Add a sidebar
gridOptions = gb.build()

AgGrid(
    map_table,
    gridOptions=gridOptions,
    #theme="light",
    height=350,
    width='100%'
)

## Appendix

st.markdown('You can view the code on my [GitHub](https://duckduckgo.com)')
