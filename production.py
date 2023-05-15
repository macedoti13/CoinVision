from functions import *
import os
import subprocess
import pickle

def main():
    model_filename = 'coin_classifier.pkl'
    training_data_filename = 'training_data.pkl'
    
    labels = [
        ['1_real'],
        ['50_cents'],
        ['25_cents'],
        ['5_cents'],
        ['10_cents'],
        ['50_cents', '25_cents', '1_real', '5_cents', '10_cents'],
        ['50_cents', '25_cents', '10_cents'],
        ['1_real', '50_cents'],
        ['1_real', '50_cents', '10_cents'],
        ['25_cents', '10_cents']
    ]

    # Check if the model file exists
    if not os.path.exists(model_filename):
        # If the model file does not exist, collect all coins and create training data
        training_data = []

        for i in range(10):
            img_gray, img_rgb = prepare_img(f'images/img{i}.png')
            circles = find_circles(img_gray)
            circles = treat_circles(circles)
            
            if circles is not None:
                filtered_circles = filter_inner_circles(circles, 10)
                coins_histograms = create_histogram_of_visual_words(img_rgb, filtered_circles)
                update_training_data(coins_histograms, labels[i], training_data)
        
        save_training_data(training_data, training_data_filename)

        # Run the train_classifier.py script to train the model
        subprocess.run(['python', 'train_classifier.py'])
    
    # Load the trained model
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    for i in range(10):
        img_gray, img_rgb = prepare_img(f'images/img{i}.png')
        circles = find_circles(img_gray)
        circles = treat_circles(circles)
        
        if circles is not None:
            filtered_circles = filter_inner_circles(circles, 10)
            coins_histograms = create_histogram_of_visual_words(img_rgb, filtered_circles)

            # Predict the class of each coin
            predictions = model.predict(coins_histograms)
            
            print(f'Predictions for image {i}: {predictions}')

main()