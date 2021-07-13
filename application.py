
import jinja2
import os
import cv2
import glob

from flask import Flask, request, render_template
import tensorflow.keras as keras
from artclassifier.artclassifier import art_style_classifier
from artclassifier.utils import cub,exp,imp,pop,art_map,get_model

app = Flask(__name__) 
app.jinja_env.filters['zip'] = zip


@app.route('/') 
def homepage():
    return render_template('homepage.html')

@app.route('/about') 
def about():
    return render_template('about.html')

@app.route('/classifier') 
def classifier():
    return render_template('classifier.html')

@app.route('/artstyles', methods=['POST']) 
def artstyles():
    if request.method == 'POST':
        image = request.files['file'] 
        print(image)
        name = ['/img/'+image.filename]
        print(name[0])
        image.save('./static'+name[0])
        #npimg = np.fromfile(request.files['file'], np.uint8)
        #npimg_resized = np.resize(npimg,(224,224))
        #img = cv2.cvtColor(npimg_resized, cv2.COLOR_BGR2RGB)
        image = keras.preprocessing.image.load_img(f'./static{name[0]}',target_size=(224,224))
        styles = ['Cubism','Expressionism','Impressionism','PopArt']
        model = get_model()
        print(model)
        prediction = art_style_classifier(image,model,styles)
        style_imgs = []
        for style in prediction.columns:
            style_img = f'{style}.png'
            style_imgs.append(style_img)
        
        return render_template('artstyles.html', name = name,prediction=prediction,
                                style_imgs=style_imgs,zip=zip)

@app.route('/galerie') 
def galerie():
    source_path = './static/galerie'
    painting_files = [f for f in os.listdir(source_path) if f.endswith('.png')]
    #print(painting_files)
    paths=[]
    for file in painting_files:
        path = 'galerie/'+ file
        paths.append(path)
    print(paths)
    return render_template('galerie.html',paths=paths)

@app.route('/contact') 
def contact():
    return render_template('contact.html')

@app.route('/Impressionism')
def impressionism():
    map = art_map(imp)
    #return map._repr_html_()
    map.save('templates/imp_map.html')
    return render_template('impressionism.html')

@app.route('/Expressionism')
def expressionism():
    map = art_map(exp)
    map.save('templates/exp_map.html')
    return render_template('expressionism.html')

@app.route('/Cubism')
def cubism():
    map = art_map(cub)
    map.save('templates/cub_map.html')
    return render_template('cubism.html')

@app.route('/PopArt')
def popart():
    map = art_map(pop)
    map.save('templates/pop_map.html')
    return render_template('popart.html')



if __name__ == "__main__":
    # runs app and debug=True ensures that when we make changes the web server restarts
    app.run(debug=True)
