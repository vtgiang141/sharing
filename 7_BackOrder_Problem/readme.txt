ACKNOWLEDGEMENTS

    The dataset used in this question is based on a dataset by tiredgeek.
    we can down load at: https://drive.google.com/file/d/0By3HmHsFxk5Oeks3THpWU3NVVUk/view 

THE PROGRAM OPERATION

    + Tree folder:
	    back_order_problem
            ├───code
            │   ├───.ipynb_checkpoints
            │   └───test
            │       ├───bin
            │       ├───include
            │       └───lib
            │           └───python3.6
            ├───dataset
            ├───document
            ├───input
            ├───model
            └───output

    + Create a virtual environment by the terminal, then direct to code folder in terminal and type "pip3 install -r requirement.txt".
    This helps you can create the environment for this project more quickly
    + In the terminal, type "sudo su root" -> "source name_env/bin/activate" -> jupyter notebook --allow-root

    + The program have 5 files in code folder what operate all of processes:
        1. File 01_Preprocessing_data.ipynb: is to reprocessing data, to access nan values, missing data, outlier,...
        2. File 02_EDA_DataStory_BackOrder.ipynb: is to evaluate features, decide that what will be keep or notebook
        3. File 03_Modelling.ipynb: run many training algorithms in a small dataset to save the time, then choose the best model, tuning it 
        to have the best parametters of 03_Modelling
        4. 04_Tuning_Model.ipynb: is to tune parametters and optimize parametters models

        5. testing.ipynb: is to test with new data, new data will be contained in input folder, when you run testing file in jupyter notebook,
        implementing the main() function, the predicted results will be saved to output folder

    + All documents relate to this project are contained in document folder
   
