from functions import *
import os
import subprocess
import pickle
import matplotlib.pyplot as plt

def main():
    model_filename = 'files/coin_classifier.pkl'
    training_data_filename = 'files/training_data.pkl'
    log_filename = 'files/log.txt'

    labels = [
        ['1 real'],
        ['50 centavos'],
        ['25 centavos'],
        ['5 centavos'],
        ['10 centavos'],
        ['50 centavos', '25 centavos', '1 real', '5 centavos', '10 centavos'],
        ['50 centavos', '25 centavos', '10 centavos'],
        ['1 real', '50 centavos'],
        ['1 real', '50 centavos', '10 centavos'],
        ['25 centavos', '10 centavos']
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

    # Images for later plotting
    images = []
    
    # Open the log file
    with open(log_filename, 'w') as log_file:

        for i in range(10):
            img_gray, img_rgb = prepare_img(f'images/img{i}.png')
            circles = find_circles(img_gray)
            circles = treat_circles(circles)
            
            if circles is not None:
                filtered_circles = filter_inner_circles(circles, 10)
                coins_histograms = create_histogram_of_visual_words(img_rgb, filtered_circles)

                # Predict the class of each coin
                predictions = model.predict(coins_histograms)
            
                # Draw the circles and predictions
                for (x, y, r), prediction in zip(filtered_circles, predictions):
                    cv2.circle(img_rgb, (x, y), r, (0, 255, 0), 8)
                    cv2.putText(img_rgb, prediction, (x-100, y-(r+30)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8)
                
                # Calculate the total value of the coins in the image
                total_value = calculate_total_value(predictions)

                # Get the conversion rates for other currencies
                currencies = ['USD', 'EUR', 'GBP', 'ARS', 'CNY', 'CAD']
                conversion_rates = get_conversion_rates(currencies)

                # Write the total value and conversion rates to the log file
                log_file.write(f"Image {i+1}: {total_value} reais são equivalentes a: \n")
                
                for currency, rate in conversion_rates.items():
                    converted_value = total_value * rate
                    log_file.write(f"   - {round(converted_value, 2)} {currency}\n")
                    
                log_file.write("\n")
                
                # Store the image for later plotting
                images.append(img_rgb)

    # Plotting starts after all log file writing is done
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    # Display all images
    for i, img_rgb in enumerate(images):
        axs[i // 5, i % 5].imshow(img_rgb)
        axs[i // 5, i % 5].axis('off')  # to hide x, y axes

    # Display all subplots
    plt.tight_layout()
    plt.show()

main()
