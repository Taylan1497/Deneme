# Deneme
To see original plots please first check the Plots directory, if you run scripts before that you will replace your results with original ones. Thats why, I created a backup folder containing these results.

All python scripts and data folder should be in the same directory.

To train the model and get accuracy and validation accuracy results and the plot of it please run

python Main.py

This will create train validation plot in the Plots Directory. This also will save the model as named “new_Model_Apr3.pk”, this pk file will be also in the same directory.

To evaluate test set and see next word prediction given in the 6)b, please run:

python Eval.py

To see the embeddings and 2d plot of it please run

python tsne.py 

This will create plot in the Plots directory.

References are given in the report.
