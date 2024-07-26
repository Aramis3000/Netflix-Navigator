<h1 align="center">Netflix-Navigator</h1>
<p><font size="3">
A Netflix Recommeding web-app for a series/movies based on list of entered choices of movies/series in your preferred language. using 
<ul> 
<li>Backend: Python, Flask</li>
<li>Frontend: HTML, CSS, JavaScript</li>
<li>Data Processing : Pandas, Scikit-learn</li>
</ul>
</p>

 # This web-app contains 3 main pages:
- [Home Page](#home-page)
- [Recommendation Page](#recommendation-page)
- [Movie Detail Page](#movie-detail-page)
- [Netflix Page](#netflix-page)

## Home Page
Here the user can choose list of their favourite movies and series and their preferred language. For example, I have entered a list with 3 Action Movies(Iron Man, Batman Begins and Avengers), an animated series(Spider-Man Into the SpiderVerse) and a drama series(Narcos) as my list of choices and English as my preferred language.
Clicking on the Get Started button the user will see the list of recommendations.
![](/app/static/screenshots/Screenshot-Home.png)

## Recommendation Page
Here the user will get poster images of all the recommended movies and series sorted based upon their IMDb Scores.
![](/app/static/screenshots/Screenshot-RecommendationPage1.png)
![](/app/static/screenshots/Screenshot-RecommendationPage2.png)

Clicking on any poster image, the user will be sent to the Movie Details page for the corresponding title.

## Movie Detail Page
Here are the complete details of the user selected title like Genre, Movie Summary, Languages in which movie is available, IMDb scores, Directors, Writers and Actors and so on. User will also find a link at the end of the page for the Netflix Page of the corresponding title. 
![](/app/static/screenshots/Screenshot-MovieDetailPage.png)

# How To Use

To be able to use this web app locally in a development environment you will need the following:

1) Make sure you have Git and Flask installed on your device.

2) Ensure that you are in the 'Netflix-Navigator/app' directory

3) You can run the Netflix App using the following command from your terminal:

```
# Set the FLASK_APP environment
set FLASK_APP=app.py

#Run Flask
flask run
```
