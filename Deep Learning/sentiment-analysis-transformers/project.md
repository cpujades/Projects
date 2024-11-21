# Text Translation and Sentiment Analysis using Transformers

## Project Overview:

The objective of this project is to analyze the sentiment of movie reviews in three different languages - English, French, and Spanish. We have been given 30 movies, 10 in each language, along with their reviews and synopses in separate CSV files named `movie_reviews_eng.csv`, `movie_reviews_fr.csv`, and `movie_reviews_sp.csv`.

- The first step of this project is to convert the French and Spanish reviews and synopses into English. This will allow us to analyze the sentiment of all reviews in the same language. We will be using pre-trained transformers from HuggingFace to achieve this task.

- Once the translations are complete, we will create a single dataframe that contains all the movies along with their reviews, synopses, and year of release in all three languages. This dataframe will be used to perform sentiment analysis on the reviews of each movie.

- Finally, we will use pretrained transformers from HuggingFace to analyze the sentiment of each review. The sentiment analysis results will be added to the dataframe. The final dataframe will have 30 rows


The output of the project will be a CSV file with a header row that includes column names such as **Title**, **Year**, **Synopsis**, **Review**, **Review Sentiment**, and **Original Language**. The **Original Language** column will indicate the language of the review and synopsis (*en/fr/sp*) before translation. The dataframe will consist of 30 rows, with each row corresponding to a movie.


```python
# imports
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline, AutoTokenizer, AutoModel
```

### Get data from `.csv` files and then preprocess data


```python
# TODO 1: use the `pd.read_csv()` function to read the movie_review_*.csv files into 3 separate pandas dataframes

# Note: All the dataframes would have different column names. For testing purposes
# you should have the following column names/headers -> [Title, Year, Synopsis, Review]

def preprocess_data() -> pd.DataFrame:
    """
    Reads movie data from .csv files, map column names, add the "Original Language" column,
    and finally concatenate in one resultant dataframe called "df".
    """
    df_en = pd.read_csv("data/movie_reviews_eng.csv", header=0, names=["Title", "Year", "Synopsis", "Review"])
    df_en['Original Language'] = 'en'
    df_fr = pd.read_csv("data/movie_reviews_fr.csv", header=0, names=["Title", "Year", "Synopsis", "Review"])
    df_fr['Original Language'] = 'fr'
    df_es = pd.read_csv("data/movie_reviews_sp.csv", header=0, names=["Title", "Year", "Synopsis", "Review"])
    df_es['Original Language'] = 'es'
    
    df = pd.concat([df_en, df_fr, df_es], ignore_index=True)
    
    return df

df = preprocess_data()
```


```python
df.tail(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Le Dîner de Cons</td>
      <td>1998</td>
      <td>Le film suit l'histoire d'un groupe d'amis ric...</td>
      <td>"Je n'ai pas aimé ce film du tout. Le concept ...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>16</th>
      <td>La Tour Montparnasse Infernale</td>
      <td>2001</td>
      <td>Deux employés de bureau incompétents se retrou...</td>
      <td>"Je ne peux pas croire que j'ai perdu du temps...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Astérix aux Jeux Olympiques</td>
      <td>2008</td>
      <td>Dans cette adaptation cinématographique de la ...</td>
      <td>"Ce film est une déception totale. Les blagues...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Les Visiteurs en Amérique</td>
      <td>2000</td>
      <td>Dans cette suite de la comédie française Les V...</td>
      <td>"Le film est une perte de temps totale. Les bl...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Babylon A.D.</td>
      <td>2008</td>
      <td>Dans un futur lointain, un mercenaire doit esc...</td>
      <td>"Ce film est un gâchis complet. Les personnage...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Roma</td>
      <td>2018</td>
      <td>Cleo (Yalitza Aparicio) es una joven empleada ...</td>
      <td>"Roma es una película hermosa y conmovedora qu...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>21</th>
      <td>La Casa de Papel</td>
      <td>(2017-2021)</td>
      <td>Esta serie de televisión española sigue a un g...</td>
      <td>"La Casa de Papel es una serie emocionante y a...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Y tu mamá también</td>
      <td>2001</td>
      <td>Dos amigos adolescentes (Gael García Bernal y ...</td>
      <td>"Y tu mamá también es una película que se qued...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>23</th>
      <td>El Laberinto del Fauno</td>
      <td>2006</td>
      <td>Durante la posguerra española, Ofelia (Ivana B...</td>
      <td>"El Laberinto del Fauno es una película fascin...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Amores perros</td>
      <td>2000</td>
      <td>Tres historias se entrelazan en esta película ...</td>
      <td>"Amores perros es una película intensa y conmo...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Águila Roja</td>
      <td>(2009-2016)</td>
      <td>Esta serie de televisión española sigue las av...</td>
      <td>"Águila Roja es una serie aburrida y poco inte...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Toc Toc</td>
      <td>2017</td>
      <td>En esta comedia española, un grupo de personas...</td>
      <td>"Toc Toc es una película aburrida y poco origi...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>27</th>
      <td>El Bar</td>
      <td>2017</td>
      <td>Un grupo de personas quedan atrapadas en un ba...</td>
      <td>"El Bar es una película ridícula y sin sentido...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Torrente: El brazo tonto de la ley</td>
      <td>1998</td>
      <td>En esta comedia española, un policía corrupto ...</td>
      <td>"Torrente es una película vulgar y ofensiva qu...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>29</th>
      <td>El Incidente</td>
      <td>2014</td>
      <td>En esta película de terror mexicana, un grupo ...</td>
      <td>"El Incidente es una película aburrida y sin s...</td>
      <td>es</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (30, 5)



### Text translation

Translate the **Review** and **Synopsis** column values to English.


```python
# load translation models and tokenizers
# TODO 2:
fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
es_en_model_name = "Helsinki-NLP/opus-mt-es-en"
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)
es_en_model = MarianMTModel.from_pretrained(es_en_model_name)
fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
es_en_tokenizer = MarianTokenizer.from_pretrained(es_en_model_name)

# TODO 3: Complete the function below
def translate(text: str, model, tokenizer) -> str:
    """
    function to translate a text using a model and tokenizer
    """
    # encode the text using the tokenizer
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # generate the translation using the model
    outputs = model.generate(**inputs)

    # decode the generated output and return the translated text
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded
```

    /Users/carlospujades/.pyenv/versions/3.11.0/envs/udacity_dl/lib/python3.11/site-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")



```python
fr_reviews = str(df[df['Original Language'] == 'fr']['Review'].values)
fr_reviews
```




    '[\'"La La Land est un film absolument magnifique avec des chansons qui restent dans la tête pendant des jours. Les acteurs sont incroyables et leur alchimie est palpable. Les scènes de danse sont absolument éblouissantes et l\\\'histoire est touchante et authentique."\'\n \'"Intouchables est un film incroyablement touchant avec des acteurs incroyables et une histoire inspirante. Les blagues sont intelligentes et jamais offensantes, et l\\\'émotion est parfaitement dosée. C\\\'est un film qui vous fera rire et pleurer, et qui vous rappellera l\\\'importance de l\\\'amitié et de la compassion."\'\n \'"Amélie est un film absolument charmant qui vous fera sourire du début à la fin. L\\\'esthétique du film est magnifique et imaginative, et la musique est enchanteresse. Audrey Tautou est incroyablement charismatique dans le rôle-titre, et l\\\'histoire est pleine de moments touchants et de personnages inoubliables."\'\n \'"Les Choristes est un film magnifique qui vous fera rire et pleurer. La musique est absolument émouvante et les performances sont incroyables, en particulier celle des jeunes acteurs. L\\\'histoire est touchante et universelle, et elle montre l\\\'importance de l\\\'art et de l\\\'éducation pour transformer des vies."\'\n \'"Le Fabuleux Destin d\\\'Amélie Poulain est un film absolument charmant qui vous fera sourire du début à la fin. L\\\'esthétique du film est magnifique et imaginative, et la musique est enchanteresse. Audrey Tautou est incroyablement charismatique dans le rôle-titre, et l\\\'histoire est pleine de moments touchants et de personnages inoubliables."\'\n \'"Je n\\\'ai pas aimé ce film du tout. Le concept de rire des gens qui sont considérés comme des idiots est offensant et le film n\\\'a pas réussi à me faire rire. Les personnages sont tous désagréables et l\\\'histoire est ennuyeuse."\'\n \'"Je ne peux pas croire que j\\\'ai perdu du temps à regarder cette absurdité. Le film est plein de blagues stupides et de scènes qui sont tout simplement inutiles. Les personnages sont irritants et l\\\'histoire est ridicule. Évitez ce film à tout prix."\'\n \'"Ce film est une déception totale. Les blagues sont sans intérêt et les acteurs semblent ne pas savoir quoi faire avec leurs personnages. Les effets spéciaux sont mauvais et l\\\'histoire est plate et prévisible. Je recommande de ne pas gaspiller votre temps à regarder ce film."\'\n \'"Le film est une perte de temps totale. Les blagues sont répétitives et ennuyeuses, les personnages sont caricaturaux et irritants, et l\\\'histoire est stupide et incohérente. Je ne recommande pas ce film à qui que ce soit."\'\n \'"Ce film est un gâchis complet. Les personnages sont sans intérêt, l\\\'intrigue est confuse et les effets spéciaux sont mauvais. Le film essaie d\\\'être intelligent et cool, mais échoue lamentablement. Évitez ce film à tout prix."\']'




```python
# TODO 4: Update the code below

# Translate each French review individually
fr_reviews = df[df['Original Language'] == 'fr']['Review']
fr_reviews_en = fr_reviews.apply(translate, model=fr_en_model, tokenizer=fr_en_tokenizer)

# Translate each French synopsis individually
fr_synopsis = df[df['Original Language'] == 'fr']['Synopsis']
fr_synopsis_en = fr_synopsis.apply(translate, model=fr_en_model, tokenizer=fr_en_tokenizer)

# Translate each Spanish review individually
es_reviews = df[df['Original Language'] == 'es']['Review']
es_reviews_en = es_reviews.apply(translate, model=es_en_model, tokenizer=es_en_tokenizer)

# Translate each Spanish synopsis individually
es_synopsis = df[df['Original Language'] == 'es']['Synopsis']
es_synopsis_en = es_synopsis.apply(translate, model=es_en_model, tokenizer=es_en_tokenizer)

# Make a copy of the dataframe to store translations
df_translated = df.copy()

# Update dataframe with translated text
df_translated.loc[df['Original Language'] == 'fr', 'Review'] = fr_reviews_en
df_translated.loc[df['Original Language'] == 'fr', 'Synopsis'] = fr_synopsis_en
df_translated.loc[df['Original Language'] == 'es', 'Review'] = es_reviews_en
df_translated.loc[df['Original Language'] == 'es', 'Synopsis'] = es_synopsis_en
```


```python
df_translated.tail(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Le Dîner de Cons</td>
      <td>1998</td>
      <td>The film follows the story of a group of rich ...</td>
      <td>"I didn't like this movie at all. The concept ...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>16</th>
      <td>La Tour Montparnasse Infernale</td>
      <td>2001</td>
      <td>Two incompetent office workers find themselves...</td>
      <td>"I can't believe I've wasted time watching thi...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Astérix aux Jeux Olympiques</td>
      <td>2008</td>
      <td>In this film adaptation of the popular comic s...</td>
      <td>"This film is a complete disappointment. The j...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Les Visiteurs en Amérique</td>
      <td>2000</td>
      <td>In this continuation of the French comedy The ...</td>
      <td>"The film is a total waste of time. The jokes ...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Babylon A.D.</td>
      <td>2008</td>
      <td>In the distant future, a mercenary has to esco...</td>
      <td>"This film is a complete mess. The characters ...</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Roma</td>
      <td>2018</td>
      <td>Cleo (Yalitza Aparicio) is a young domestic wo...</td>
      <td>"Rome is a beautiful and moving film that pays...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>21</th>
      <td>La Casa de Papel</td>
      <td>(2017-2021)</td>
      <td>This Spanish television series follows a group...</td>
      <td>"The Paper House is an exciting and addictive ...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Y tu mamá también</td>
      <td>2001</td>
      <td>Two teenage friends (Gael García Bernal and Di...</td>
      <td>"And your mom is also a movie that stays with ...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>23</th>
      <td>El Laberinto del Fauno</td>
      <td>2006</td>
      <td>During the Spanish postwar period, Ofelia (Iva...</td>
      <td>"The Labyrinth of Fauno is a fascinating and e...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Amores perros</td>
      <td>2000</td>
      <td>Three stories intertwine in this Mexican film:...</td>
      <td>"Amores dogs is an intense and moving film tha...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Águila Roja</td>
      <td>(2009-2016)</td>
      <td>This Spanish television series follows the adv...</td>
      <td>"Red Eagle is a boring and uninteresting serie...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Toc Toc</td>
      <td>2017</td>
      <td>In this Spanish comedy, a group of people with...</td>
      <td>"Toc Toc is a boring and unoriginal film that ...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>27</th>
      <td>El Bar</td>
      <td>2017</td>
      <td>A group of people are trapped in a bar after M...</td>
      <td>"The Bar is a ridiculous and meaningless film ...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Torrente: El brazo tonto de la ley</td>
      <td>1998</td>
      <td>In this Spanish comedy, a corrupt cop (played ...</td>
      <td>"Torrente is a vulgar and offensive film that ...</td>
      <td>es</td>
    </tr>
    <tr>
      <th>29</th>
      <td>El Incidente</td>
      <td>2014</td>
      <td>In this Mexican horror film, a group of people...</td>
      <td>"The Incident is a boring and frightless film ...</td>
      <td>es</td>
    </tr>
  </tbody>
</table>
</div>



### Sentiment Analysis

Use HuggingFace pretrained model for sentiment analysis of the reviews. Store the sentiment result **Positive** or **Negative** in a new column titled **Sentiment** in the dataframe.


```python
# TODO 5: Update the code below
# load sentiment analysis model

from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
sentiment_classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0)

# TODO 6: Complete the function below
def analyze_sentiment(text, classifier):
    """
    function to perform sentiment analysis on a text using a model
    
    """
    result = classifier(text)[0]['label'].title()
    
    return result
```


```python
# TODO 7: Add code below for sentiment analysis
# perform sentiment analysis on reviews and store results in new column

df_translated['Sentiment'] = df_translated['Review'].apply(analyze_sentiment, classifier=sentiment_classifier)
```


```python
df_translated.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>La La Land</td>
      <td>2016</td>
      <td>This musical tells the story of a budding actr...</td>
      <td>"The Land is an absolutely beautiful film with...</td>
      <td>fr</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Toc Toc</td>
      <td>2017</td>
      <td>In this Spanish comedy, a group of people with...</td>
      <td>"Toc Toc is a boring and unoriginal film that ...</td>
      <td>es</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Babylon A.D.</td>
      <td>2008</td>
      <td>In the distant future, a mercenary has to esco...</td>
      <td>"This film is a complete mess. The characters ...</td>
      <td>fr</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Le Fabuleux Destin d'Amélie Poulain</td>
      <td>2001</td>
      <td>This romantic comedy tells the story of Amélie...</td>
      <td>"The Fabulous Destiny of Amélie Poulain is an ...</td>
      <td>fr</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Nice Guys</td>
      <td>2016</td>
      <td>In 1970s Los Angeles, a private eye (Ryan Gosl...</td>
      <td>"The Nice Guys tries too hard to be funny, and...</td>
      <td>en</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>0</th>
      <td>The Shawshank Redemption</td>
      <td>1994</td>
      <td>Andy Dufresne (Tim Robbins), a successful bank...</td>
      <td>"The Shawshank Redemption is an inspiring tale...</td>
      <td>en</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Les Choristes</td>
      <td>2004</td>
      <td>This film tells the story of a music teacher w...</td>
      <td>"The Choristes are a beautiful film that will ...</td>
      <td>fr</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Roma</td>
      <td>2018</td>
      <td>Cleo (Yalitza Aparicio) is a young domestic wo...</td>
      <td>"Rome is a beautiful and moving film that pays...</td>
      <td>es</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Astérix aux Jeux Olympiques</td>
      <td>2008</td>
      <td>In this film adaptation of the popular comic s...</td>
      <td>"This film is a complete disappointment. The j...</td>
      <td>fr</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Águila Roja</td>
      <td>(2009-2016)</td>
      <td>This Spanish television series follows the adv...</td>
      <td>"Red Eagle is a boring and uninteresting serie...</td>
      <td>es</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
import os

# create the result directory if it does not exist
os.makedirs("result", exist_ok=True)

# export the results to a .csv file
df_translated.to_csv("result/reviews_with_sentiment.csv", index=False)
```
