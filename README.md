# Colab-Start

Getting started with google colab

## Sections

## Notes

1. There are cells in each .ipynb (Jupyter Notebook). Click on a cell and choose to either add a
   1. text cell: contains text data such as markdown. can't be run.
   2. code cell: contains runnable code. Output is directly below the code
2. The variables in one code cell can be reused in another.
3. you can add code snippets
4. Tensorlfow:
   1. install tensorflow for cpu and gpu using:
      1. `!pip install tensorflow`
      2. `!pip install tensorflow-gpu`
   2. check version using:

        ```python
        import tensorflow as tf
        print(tf.__version__)
        ```

   3. import and unzip files using

        ```python
        !wget -cq http://www.laurencemoroney.com/wp-content/uploads/2019/02/breast-cancer-colab.zip
        !unzip -qq breast-cancer-colab.zip
        ```

   4. Load using pandas as usual

        ```python
        import pandas as pd
        from google.colab import files
        x_train = pd.read_csv("xtrain.csv", header=None)
        y_train = pd.read_csv("ytrain.csv", header=None)
        x_test = pd.read_csv("xtest.csv", header=None)
        y_test = pd.read_csv("ytest.csv", header=None)
        ```

   5. Initialize the ANN

        ```python
        from keras.models import Sequential
        from keras.layers import Dense

        # initiating the ANN
        classifier = Sequential() 

        classifier.add(Dense(units = 16, activation = 'relu', input_dim = 30))
        classifier.add(Dense(units = 8, activation = 'relu'))
        classifier.add(Dense(units = 6, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        ```

   6. Compile it 

        ```python
        classifier.compile(optimizer='rmsprop', loss='binary_crossentropy')
        ```

   7. Train the model. Make sure runtime is GPU

        ```python
        classifier.fit(x_train, y_train, batch_size=1, epochs = 100)
        ```

   8. Perform predictions

        ```python
        y_pred = classifier.predict(x_test)
        y_pred = [1 if y>=0.5 else 0 for y in y_pred]
        print(y_pred)
        ```

   9. Display statistics

        ```python
        total = 0
        correct = 0
        wrong = 0
        for i in y_pred:
        total = total+1
        if y_test.at[i,0] == y_pred[i] :
            correct = correct +1
        else:
            wrong = wrong +1
        print(f"Total : {total} \n Correct : {correct} \n Wrong : {wrong}")
        ```

5.  k
