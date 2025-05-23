from django.shortcuts import render,HttpResponse


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import tempfile
global AERFC,encoder
autoencoder=None
sc=None
# Create your views here.
global X_train_scaled,X_test_scaled
def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')


# Ignore all warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,f1_score
#pip install openpyxl
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

global X_train,X_test,y_train,y_test
X_train = None
def Upload_data(request):
    load=True
    global sc
    global X_train_scaled,X_test_scaled
    global X_train,X_test,y_train,y_test
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        df=pd.read_csv(default_storage.path(file_path))
        le = LabelEncoder()
        df['type'] = le.fit_transform(df['type'])
        df = df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',"isFraud","isFlaggedFraud"]]
        x = df.drop(['isFraud'],axis = 1)
        y = df['isFraud']
        from sklearn.utils import resample
        x, y = resample(x, y)
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)
        default_storage.delete(file_path)
        print('---done---')
        outdata=df.head(100)
        return render(request,'prediction.html',{'predict':outdata.to_html()})
    return render(request,'prediction.html',{'upload':load})

labels=['Fraud','NotFraud']
precision = []
recall = []
fscore = []
accuracy = []

def PerformanceMetrics(image,algorithm, testY,predict):
    global labels
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' F1-SCORE      : '+str(f))
    
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.legend()
    plt.savefig(image)
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
def Dnn_existing(request):
    if X_train is None:
        messages.error(request,'Please upload dataset first')
        return redirect('upload')
    else:
        import os
        from sklearn.metrics import accuracy_score, classification_report
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import load_model
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.utils import to_categorical

        # Define file paths
        model_file = "model/dnn_model.h5"

        # Prepare target variable for DNN
        y_train_categorical = to_categorical(y_train)
        y_test_categorical = to_categorical(y_test)

        if os.path.exists(model_file):
            # Load the trained DNN model
            dnn = load_model(model_file)
            print("Deep Neural Network model loaded successfully.")
        else:
            # Initialize and train the DNN model
            print("Training a new Deep Neural Network model...")
            dnn = Sequential([
                Dense(128, activation='relu', input_dim=X_train.shape[1]),
                Dense(64, activation='relu'),
                Dense(y_train_categorical.shape[1], activation='softmax')  # Output layer
            ])
            dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Add early stopping to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Train the model
            dnn.fit(X_train, y_train_categorical, 
                    validation_split=0.2, 
                    epochs=100, 
                    batch_size=32, 
                    callbacks=[early_stopping], 
                    verbose=1)
            
            # Save the model
            os.makedirs("model", exist_ok=True)
            dnn.save(model_file)
            print("Model saved successfully.")

        # Predict and evaluate
        predict = dnn.predict(X_test)
        predict_classes = predict.argmax(axis=1)  # Convert probabilities to class labels

        # Convert y_test to class indices
        y_test_classes = y_test  # Ensure y_test is not categorical
        # Performance metrics
        image='static/images/DNN.png'
        PerformanceMetrics(image,'Deep Neural Network',y_test_classes, predict_classes)
    return render(request,'prediction.html',
                  {'algorithm':'Deep Neural Network',
                   'image':image,
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})


def randomforest(request):
    import os
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    model_file = "model/rfc_model.pkl"

    if os.path.exists(model_file):
        # Load the trained model
        rfc = joblib.load(model_file)
        print("Random Forest Classifier model loaded successfully.")
    else:
        # Initialize and train the RFC model
        print("Training a new Random Forest Classifier model...")
        rfc = RandomForestClassifier(n_estimators=10, random_state=1, max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=2)
        rfc.fit(X_train, y_train)
        
        # Save the model
        os.makedirs("model", exist_ok=True)
        joblib.dump(rfc, model_file)
        print("Model saved successfully.")

    # Predict and evaluate
    predict = rfc.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predict))
    image='static/images/RFC.png'
    PerformanceMetrics(image,"Random Forest Classifier", predict, y_test)
    return render(request,'prediction.html',
                  {'algorithm':'Random Forest Classifier',
                   'image':image,
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})


def prediction_view(request):
    import joblib
    Test=True
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        testdata = pd.read_csv(default_storage.path(file_path))
        test = testdata.drop(['type','step','nameOrig','nameDest','isFraud'], axis = 1)
        if sc:
            model_file = "model/rfc_model.pkl"
            rfc = joblib.load(model_file)
            test = sc.transform(test)
            predict = rfc.predict(test)
            labels=['Fraud','NotFraud']  
            list=[]  
            for i in predict:
                list.append(labels[i])
            testdata['predicted']=list
            default_storage.delete(file_path)
            return render(request,'prediction.html',{'predict':testdata.to_html()}) 
        else:
            messages.error(request,'The model is not yet loaed Please contact to admin to load model')
            return redirect('prediction')

    return render(request,'prediction.html',{'test':Test})

