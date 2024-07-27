import os
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .utils import (
    load_and_preprocess_data, 
    train_model, 
    save_model, 
    load_model, 
    make_prediction,
    perform_undersampling, 
    perform_oversampling,
    evaluate_model
)
from sklearn.model_selection import train_test_split

def input_view(request):
    return render(request, 'input.html')

def process_input(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        dataset = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save(dataset.name, dataset)
        file_path = fs.path(filename)

        try:
            # Load and preprocess data
            data = load_and_preprocess_data(file_path)

            # Perform undersampling and oversampling
            undersampled_data = perform_undersampling(data)
            X_under = undersampled_data.drop('Class', axis=1)
            y_under = undersampled_data['Class']
            X_res, y_res = perform_oversampling(X_under, y_under)

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            model = train_model(X_train, y_train)
            model_path = os.path.join(settings.MEDIA_ROOT, 'credit_card_model.pkl')
            save_model(model, model_path)

            # Evaluate model
            model = load_model(model_path)
            metrics = evaluate_model(model, X_test, y_test)

            # Make prediction
            sample_input = [-1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]
            pred_result = make_prediction(model, sample_input)

            return render(request, 'output.html', {
                'result': pred_result,
                'metrics': metrics,
                'data_head': data.head().to_html(),
                'data_description': data.describe().to_html()
            })

        except Exception as e:
            return render(request, 'output.html', {
                'error': str(e)
            })

    return redirect('input')
