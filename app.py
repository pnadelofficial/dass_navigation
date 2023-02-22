import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import plotly.express as px
from streamlit_plotly_events import plotly_events
import pickle
import plotly.io as pio
from sklearn.manifold import TSNE

tsne = TSNE(2)
pio.templates.default = 'plotly'

st.title('DASS Navigation App')
st.write("""
This application allows you to navigate through the narrative section from the DASS dataset using semantic word vectors,
obtained from [Gensim's Doc2Vec model](https://radimrehurek.com/gensim/models/doc2vec.html).
""")

@st.cache(allow_output_mutation=True)
def load_d2v():
    dass_d2v = KeyedVectors.load('dass_d2v.kv')
    dass = pd.read_csv('DASS_Narrative_updated_2.21.23.csv').dropna().reset_index(drop=True)
    dass_df = pd.read_csv('dass_df.csv')
    return dass_d2v, dass, dass_df
dass_d2v, dass, dass_df = load_d2v()
dass_df['Abby_Project_ID'] = dass.Abby_Project_ID

# filter by DASS score as well
dass_filter = st.number_input('Filter by Total DASS Score')
dass_df = dass_df.loc[dass_df.Total >= dass_filter]


fig = px.scatter(dass_df, x='x', y='y', color='Total', hover_data=['Abby_Project_ID'])
fig_selected = plotly_events(fig, select_event=True)

if st.button('Reset'):
    fig_selected = []

if len(fig_selected) > 0:
    for selected in fig_selected:
        subset = dass_df.loc[(dass_df.x == selected['x']) & (dass_df.y == selected['y'])]
        selected_from_df = dass.loc[dass.index == subset.doc_index.to_list()[0]]
        selected_nar = selected_from_df.Narrative.to_list()[0]
        st.write(f'Survey: **{selected_from_df.Abby_Project_ID.to_list()[0]}**')
        st.write(f'Stress: {selected_from_df["DASS.Stress"].to_list()[0]}')
        st.write(f'Anxiety: {selected_from_df["DASS.Anxiety"].to_list()[0]}')
        st.write(f'Depression: {selected_from_df["DASS.Depression"].to_list()[0]}')
        st.write(f"<p>{selected_nar}</p>",unsafe_allow_html=True)
        st.markdown('<hr>',unsafe_allow_html=True)
else:
    st.write('Use the select tools in the chart above to select some works.')

st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)     