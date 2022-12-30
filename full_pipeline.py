import streamlit as st
import pandas as pd
import numpy as np 
#import matplotlib.pyplot as plt
#import seaborn as sns
#import nltk 

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")



with st.container():
    st.subheader("Hi, I am Sven :wave:")
    st.title("A Data Analyst From Germany")
    st.write(
        "I am passionate about finding ways to use Python and VBA to be more efficient and effective in business settings."
    )
    
    
    
    
    
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What I do")
        st.write("##")
        st.write(
            """
            On my YouTube channel I am creating tutorials for people who:
            - are looking for a way to leverage the power of Python in their day-to-day work.
            - are struggling with repetitive tasks in Excel and are looking for a way to use Python and VBA.
            - want to learn Data Analysis & Data Science to perform meaningful and impactful analyses.
            - are working with Excel and found themselves thinking - "there has to be a better way."
            If this sounds interesting to you, consider subscribing and turning on the notifications, so you don’t miss any content.
            """
        )
        st.write("[YouTube Channel >](https://youtube.com/c/CodingIsFun)")
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

co = [input("Enter a Comment please : ")]
def full_pipeline(co):
    # Import pre-trained model
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" 
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Create an object 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    
    def polarity_scores_roberta(example):
        # Tokenize the comments
        encoded_text = tokenizer(example, return_tensors='pt')

        # input to the model 
        output = model(**encoded_text) 
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

         # create a new columns with names
        scores_dict = {
            'roberta_neg' : scores[0],
            'roberta_neu' : scores[1],
            'roberta_pos' : scores[2],
            'comment' : ' '.join(co)
        }

        #return a diect with the polarity of the comments  
        return scores_dict 




    sia = SentimentIntensityAnalyzer()
    res = {}
    for i in tqdm(co, total=len(co)):
        try:
            text = str(co)
            myid = len(co)
            vader_result = sia.polarity_scores(text)
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
            roberta_result = polarity_scores_roberta(text)
            both = {**vader_result_rename, **roberta_result}
            res[myid] = both
        except RuntimeError:
            print(f'Broke for id {myid}')
            
            
            
            
            
      
    
    res = pd.DataFrame(res).T
    neg_df = res[(res['roberta_neg'] > res['roberta_pos'])] 
    neg_df = neg_df[['comment', 'roberta_neg']]
    if len(neg_df) == 0:
        print('Sorry, the comment eather positive nor neutral !! ')
        
    else:
        #function to input the cleaning function to multi-core processing
        def clean_apply(df):
            from nltk.tokenize import word_tokenize
            import re
            import spacy
            import string
            from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
            from nltk.corpus import stopwords
            global stop_words
            stop_words = ["a","able","about","above","abst","accordance","according","accordingly","across","act","actually","added",
                      "adj","affected","affecting","affects","after","afterwards","again","against","ah","all","almost","alone",
                      "along","already","also","although","always","am","among","amongst","an","and","announce","another","any",
                      "anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately",
                      "are","aren","arent","arise","around","as","aside","ask","asking","at","auth","available","away","awfully",
                      "b","back","be","became","because","become","becomes","becoming","been","before","beforehand","begin",
                      "beginning","beginnings","begins","behind","being","believe","below","beside","besides","between","beyond",
                      "biol","both","brief","briefly","but","by","c","ca","came","can","cannot","cause","causes","certain",
                      "certainly","co","com","come","comes","contain","containing","contains","could","couldnt","d","date","did",
                      "different","do","does","doing","done","down","downwards","due","during","e","each","ed","edu","effect",
                      "eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even",
                      "ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","few","ff","fifth",
                      "first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from",
                      "further","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone",
                      "got","gotten","h","had","happens","hardly","has","have","having","he","hed","hence","her","here","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however","hundred","i","id","ie","if","im","immediate","immediately","importance","important","in","inc","indeed","index","information","instead","into","invention","inward","is","it","itd","its","itself","j","just","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","look","looking","looks","ltd","m","made","mainly","make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","shes","should","show","showed","shown","showns","shows","significant","significantly",
                       "similar","similarly","since","six","slightly","so","some","somebody","somehow","someone","somethan",
                      "something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify",
                      "specifying","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest",
                      "sup","sure","t","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that","thats",
                      "the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","thered","therefore",
                      "therein","thereof","therere","theres","thereto","thereupon","these","they","theyd","theyre","think","this",
                      "those","thou","though","thoughh","thousand","throug","through","throughout","thru","thus","til","tip","to",
                      "together","too","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un",
                      "under","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","us","use","used",
                      "useful","usefully","usefulness","uses","using","usually","v","value","various","very","via","viz","vol",
                      "vols","vs","w","want","wants","was","wasnt","way","we","wed","welcome","went","were","werent","we've","what",
                      "whatever","whats","when","whence","whenever","where","whereafter","whereas","whereby","wherein","wheres",
                      "whereupon","wherever","whether","which","while","whim","whither","who","whod","whoever","whole","whom",
                      "whomever","whos","whose","why","widely","willing","wish","with","within","without","wont","words","world",
                      "would","wouldnt","www","x","y","yes","yet","you","youd","your","youre","yours","yourself","yourselves","z",
                      "zero"'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                      'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                      'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                      'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                      'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'own', 'same', 'so', 'than', 'love', 'like', 'loved', 'good', 'great', 'soft', 'super', 'perfect', 'comfy', 'nice', 'quality', 'true', 
                      'cute', 'comfortable','l', 'xl', 'xxl', 'warm', 'perfectly', 'expected', 'fit', 'pretty', 'perfect', 'sweatshirt', 'buy', 'bought', 'lb']
        #     stop_words = stopwords.words("english")
        #     stop_words = stop_words.extend(['not', 'Not','aren','aren\'t','couldn','couldn\'t','didn','didn\'t','doesn','doesn\'t','hadn','hadn\'t','hasn','hasn\'t','haven','haven\'t','isn','isn\'t','ma','mightn','mightn\'t','mustn','mustn\'t','needn','needn\'t','shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"])

            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


            def clean_text(text):
                text = text.lower() #lower-casing
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\d+',' ',text) #removing numbers
                text = text.translate(str.maketrans('','',string.punctuation)) #removing punctuation
                text = text.lower() #lower-casing
                text = [i for i in word_tokenize(text) if i not in stop_words] #remvoving stop-words
        #         doc = nlp(' '.join(text))
                stemmer = WordNetLemmatizer()
                text = [stemmer.lemmatize(token) for token in text] #lemmatizing the reviews
                text = ' '.join(text)    
                text = text.strip() #removing white-spaces
                return text
            df['comment'] = df.comment.apply(clean_text)
            return df



        clean_apply(neg_df)





        df = pd.read_csv('/anaconda/JupyterNotebooK/DataSets/neg_df_labeled.csv', index_col=0)
        df = df.fillna('').reset_index(drop=True)

        X = df['comment']
        y = df['pred_cat']

        x = neg_df['comment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        x = vectorizer.transform(x)


        import joblib
        # joblib.dump(lo, 'LogisticRegression_model.sav')




        # load the model from disk
#         loaded_model = joblib.load('LogisticRegression_model.sav')
#         predected = loaded_model.predict(x)
#         # result = loaded_model.score(x, y_test)
#         # print(result)
#         neg_df['predected'] = predected
#         neg_df['comment'] = co
#         neg_df.drop('roberta_neg', axis = 1, inplace = True)
#         print('*'*20 , 'Logistic Regression model', '*'*20)
#         display(neg_df)
        
        
        # load the model from disk
        loaded_model = joblib.load('GradientBoostingClassifier_model.sav')
        predected = loaded_model.predict(x)
#         result = loaded_model.score(x, y_test)
#         print(result)
        neg_df['predected'] = predected
        neg_df['comment'] = co
        print('')
        print('*'*20 , 'Gradient Boosting Classifier model', '*'*10)
        display(neg_df)

full_pipeline(co)
