# Movie-Review-Sentiment-Classification-Using-CNN

This project focuses on building a convolutional neural network (CNN) to classify movie reviews from the IMDB dataset as positive or negative. The workflow starts by downloading and extracting the dataset, followed by loading and shuffling the data to ensure randomness.<br/>Text preprocessing includes : <br/> tokenizing the reviews and padding sequences to a fixed length for consistent input size. The CNN model comprises embedding, convolutional, pooling, dropout, and dense layers, ending with a sigmoid activation for binary classification. The model is compiled using the Adam optimizer and binary cross-entropy loss. <br/>After training, the model achieves an accuracy of approximately 88% on the test set. Predictions are decoded for qualitative assessment, and detailed evaluation metrics, including precision, recall, F1-score, and a confusion matrix, are provided to gauge performance.

#### Dataset Reference
The IMDB dataset used in this project was developed by Andrew L. Maas et al. (2011) and is publicly available at:
https://ai.stanford.edu/~amaas/data/sentiment/

