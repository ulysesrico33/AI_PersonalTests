"""
24-ene-2021
Para mis pruebas usaré una lista de documentos pequeños
Sobre todo para comprobar los cálculos manualmente, una vez seguro
de los procedimientos, entonces funciona a gran escala
"""

from scipy.linalg import svd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

doc1='Machine learning es genial'
doc2='Me gusta programar en Machine Learning'
doc3='Python es padre en Machine Learning'
doc4='pipo'
corpus=[doc1,doc2,doc3,doc4]

#En TF IDF el Ln(N/#doc where word appears) es Ln (logaritmo natural).
#Para este ejemplo , para la palabra "comprobar" el valor de TF IDF (cálculo manual) es : 0.69314718

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print('TF-IDF vector')
print(df)
