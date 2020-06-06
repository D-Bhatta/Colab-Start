# Colab-Start

Getting started with google colab

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

5. To use a gpu:
    1. go to Runtime in menu bar
    2. Choose 'Change Runtime Type'
    3. Choose 'GPU' as 'Hardware Accelerator'
    4. Click save
6. Check whether GPU is being used, to save time

    ```python
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
    raise SystemError("GPU device not found")
    print(f'Found GPU at: {device_name}')
    ```

7. To use only GPU, use the `with tf.device('/gpu:0'):` code

    ```python
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    with tf.device('/gpu:0'):
    model = tf.keras.models.Sequential((
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(x_train, y_train,epochs=5)
    model.evaluate(x_test, y_test)
    ```

8. To use only CPU, use the `with tf.device('/cpu:0'):` code

    ```python
    with tf.device('/cpu:0'):
    model = tf.keras.models.Sequential((
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(x_train, y_train,epochs=5)
    model.evaluate(x_test, y_test)
    ```

9. See all local devices with

    ```python
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()
    ```

10. Checkout all local information with

    ```python
    !cat /proc/cpuinfo
    !cat /proc/meminfo
    ```

11. To upload files from mounted google drive
    1. Write the following code and navigate to given url

        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```

    2. Load fiiles

        ```python
        with open('/content/drive/My Drive/foo.txt', 'w') as f:
        f.write('Hello Google Drive!')
        !cat /content/drive/My\ Drive/foo.txt
        ```

    3. Unmount drive and save stuff

        ```python
        drive.flush_and_unmount()
        print('All changes made in this colab session should now be visible in Drive.')
        ```

12. TO manually upload and download files
    1. upload

        ```python
        from google.colab import files

        uploaded = files.upload()

        for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))
        ```

    2. download

       ```python
        from google.colab import files

        with open('example.txt', 'w') as f:
        f.write('some content')

        files.download('example.txt')
        ```
