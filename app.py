import streamlit as st
from recommendation_system import df, recommend

st.title("Movie Recommendation System")
st.write("Pick a movie you like and this will suggest similar ones based on the plot and genre.")

movie_list = df["title"].values
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    st.write("Movies similar to", selected_movie)
    results = recommend(selected_movie, top_n=5)

    if len(results) == 0:
        st.write("couldn't find recommendations for this one, try another movie")
    else:
        for movie in results:
            st.write("-", movie)
