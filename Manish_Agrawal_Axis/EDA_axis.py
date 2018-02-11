#Standard imports for Exploratory Data Analysis
import math
import pandas as pd
import numpy as np
import seaborn as sns
import json


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.plotly as py
import plotly.graph_objs as go

from datetime import date, timedelta
from wordcloud import WordCloud
from skimage.draw import ellipse
from textwrap import wrap
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import HTML, Image
from plotly import tools

plt.style.use('seaborn-white')
sns.set_context(rc={"figure.figsize": (9, 7)})

#importing the given dataset USvideos.csv 
columns = ['video_id', 'title', 'channel_title', 'category_id',
          'tags', 'views', 'likes', 'dislikes', 'comment_total',
          'thumbnail_link', 'date']

us_vid_df =  pd.read_csv("USvideos.csv", usecols = columns)

us_vid_df = us_vid_df[:][:2799]

print (us_vid_df['date'].unique())
#removing the discrepancies in the date
us_vid_df.loc[us_vid_df['date'] == '24.09xcaeyJTx4Co', 'date'] = '24.09'
us_vid_df.loc[us_vid_df['date'] == '26.0903jeumSTSzc', 'date'] = '26.09'

print (us_vid_df['date'].unique())


#formatting the date
us_vid_df['date'] = us_vid_df['date'].apply(lambda x: pd.to_datetime(str(x).replace('.','')+"2017",format='%d%m%Y'))
#print (us_vid_df['date'].unique())
#us_vid_df.loc[us_vid_df['date'] == '2017-09-132017', 'date'] = '2017-09-13'
us_vid_df['date'] = us_vid_df['date'].dt.date

us_vid_df = us_vid_df.drop_duplicates()
print (us_vid_df['date'].unique())


# dataframe to see videos per each date
def quick_insight(df, country, color):
    # dataframes for videos per each date
    vid_check = df[['video_id', 'date']].copy().groupby('date', as_index = False).count()
    vid_check.columns = ['Dates', 'Videos per date']
    
    # dataframes for 
    dates_per_id = df[['video_id', 'date']].groupby('video_id', as_index = False).count()
    dates_per_vids = dates_per_id.groupby('date', as_index = False).count()
    dates_per_vids.columns = ['Quantity of dates per video', 'Quantity of videos in a date group']
    max_days = max(dates_per_vids['Quantity of dates per video'].values)
    
    # videos appeared in database as at 13 September 2017
    sept_13_id = df.loc[df['date'] == date(2017,9,13), 'video_id'].tolist()
    sept_13 = df.loc[df['video_id'].isin(sept_13_id), ['video_id', 'date']]
    sept_13 = sept_13.groupby('date', as_index=False).count()   
    
    # combined plot
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
    
    # plotting videos per each date
    ax1 = plt.subplot(gs[0,0])
    ax1 = sns.barplot(x="Dates", y="Videos per date", data=vid_check, color='green', alpha=0.8)
    ax1.set_ylabel('Videos per date', fontsize=14)
    ax1.set_xticklabels(vid_check['Dates'].unique(), rotation=45)
    ax1.set_xlabel('')
    ax1.set_title('Videos per each date', fontsize=15)
    
    # plotting buckets of videos by quantity of trending dates
    ax2 = plt.subplot(gs[0,1])
    ax2 = sns.barplot(x="Quantity of dates per video",
                      y="Quantity of videos in a date group",
                      data=dates_per_vids, color=color)
    ax2.set_ylabel('Quantity of videos in a date group', fontsize=14)
    ax2.set_xlabel('Quantity of trending days in a bucket', fontsize=14)
    ax2.set_title('Buckets of videos by quantity of trending dates', fontsize=13)
    
    # plotting story of videos that appeared on September 13
    ax3 = plt.subplot(gs[0,2])    
    ax3 = sns.barplot(x='date', y="video_id", data=sept_13, color=color, alpha = 0.7)
    ax3.set_ylabel('Quantity of videos per date', fontsize=14)
    ax3.set_xticklabels(sept_13['date'], rotation=45)
    ax3.set_xlabel('')
    ax3.set_title('Videos started at 13 September', fontsize=15)
    
    plt.show()


#function to analyze the longest trending videos
def best_survivors(df, days, country):
    # videos with selected lifetime in top 200
    dates_per_id = df[['video_id', 'date']].groupby('video_id', as_index = False).count()
    long_surv_list = dates_per_id.loc[dates_per_id['date'] == days,'video_id'].tolist()
    long_surv_vid = df.loc[df['video_id'].isin(long_surv_list),
                           ['title','date','views','likes','dislikes','comment_total']]
    long_surv_vid['views'] = long_surv_vid['views'].apply(lambda x: x/1000000)
    titles_list = long_surv_vid['title'].unique().tolist()   

    # plotting videos views history
    views = []
    likes = []
    dislikes = []
    comments = []
    plots_list = [views, likes, dislikes, comments]
    column_list = ['views', 'likes', 'dislikes', 'comment_total']
    boolean_list = [False, False, False, True]
    colors_list = []
    for i in range(0,len(titles_list)):
        color = 'rgb('+str(np.random.randint(1,256))+","+str(np.random.randint(1,256))+","+str(np.random.randint(1,256))+")"
        colors_list.append(color)
    
    for x in range (0,len(plots_list)):
        for i in range (0, len(titles_list)):
            vt = titles_list[i]
            trace = go.Scatter(x = long_surv_vid.loc[long_surv_vid['title']== vt,'date'],
                                y = long_surv_vid.loc[long_surv_vid['title']== vt, column_list[x]],
                                name = vt,
                                line = dict(width = 2, color = colors_list[i]),
                                legendgroup = vt,
                                showlegend = boolean_list[x])
    
            plots_list[x].extend([trace])
    
    fig = tools.make_subplots(rows=2, cols=2, 
                              subplot_titles=('Views', 'Comments', 'Likes', 'Dislikes'))
    
    for i in views:
        fig.append_trace(i, 1, 1)
    for i in comments:
        fig.append_trace(i, 1, 2)
    for i in likes:
        fig.append_trace(i, 2, 1)
    for i in dislikes:
        fig.append_trace(i, 2, 2)
    
    fig['layout']['xaxis1'].update(title='')
    fig['layout']['xaxis2'].update(title='')
    fig['layout']['xaxis3'].update(title='')
    fig['layout']['xaxis4'].update(title='')

    fig['layout']['yaxis1'].update(title='mln. views')
    fig['layout']['yaxis2'].update(title='comments')
    fig['layout']['yaxis3'].update(title='likes')
    fig['layout']['yaxis4'].update(title='dislikes')
    
    fig['layout'].update(width=800, height=1400)
    fig['layout'].update(title='Different metrics for videos trended for {} days'.format( days))
    fig['layout'].update(legend = dict(x=0.0, y=-0.5))

    iplot(fig, filename='customizing-subplot-axes')


#first function
quick_insight(us_vid_df, "US", 'red')

#second funciton 
best_survivors(us_vid_df, 7, "US")
