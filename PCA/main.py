#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import nltk
import pandas as pd
from PCA import preprocessing, dataloader, utils
from PCA.models import PCA

nltk.download('stopwords')


def run_model(embedding_type, x_train, y_train, x_test, y_test, train_ratio=0.8, score_iteration_limit=2):
    results = []
    print("Transformed train data shape: ", x_train.shape)
    print("Transformed test data shape: ", x_test.shape)
    print('Evaluating {} on log type {} with train/test ratio {}/{}::'.format('PCA', log_type,
                                                                              train_ratio, (1 - train_ratio)))
    model = PCA()
    model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

    scores = model.get_scores(x_test)
    fpr, tpr, thresh = utils.roc(y_test, scores)
    for i, t in enumerate(thresh):
        print("Running threshold {}".format(t))
        model.threshold = t
        acc, precision, recall, f1 = model.evaluate_scores(scores, y_test)
        benchmark_results.append([log_type, embedding_type, 'PCA', train_ratio, t, acc, precision, recall, f1])
        if i == score_iteration_limit:
            break

    return results


if __name__ == '__main__':
    # The benchmark datasets, ajust these paths
    struct_logs = {'BGL': '../data/BGL/BGL_5k.csv', 'thunder-bird': '../data/tbird/tbird2_5k.csv',
                   'spirit': '../data/spirit/spirit2_5k.csv'}
    # Paths to logsy embeddings, adjust this, need to generated first
    logsy_embeddings = {'BGL': ('../data/BGL/pca_dimensions.csv',
                                '../data/BGL/pca_dimensions_test.csv',
                                '../data/BGL/bgl_target.npy'),
                        'spirit': ('../data/spirit/pca_dimensions_spirit_train.csv',
                                   '../data/spirit/pca_dimensions_spirit_test.csv',
                                   '../data/spirit/spirit_labels.npy'),
                        'tbird': ('../data/tbird/pca_dimensions_thunderbird_train.csv',
                                  '../data/tbird/pca_dimensions_thunderbird_test.csv',
                                  '../data/tbird/tbird_labels.npy')
                        }
    # Train ratios
    train_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]

    # All combinations of struct_logs and train_ratios
    benchmark_results = []
    # Run tf-idf
    for log_type, train_ratio in itertools.product(struct_logs, train_ratios):
        (x_tr, y_train), (x_te, y_test) = dataloader.load_HPC_log(struct_logs[log_type], train_ratio=train_ratio)
        feature_extractor = preprocessing.FeatureExtractorTFIDF()
        x_train = feature_extractor.fit_transform(x_tr)
        x_test = feature_extractor.transform(x_te)
        benchmark_results += run_model('tf-idf', x_train, y_train.to_numpy(), x_test, y_test.to_numpy(),
                                       train_ratio=train_ratio)
    # run with logsy embeddings
    for log_type, (train_file, test_file, target_file) in logsy_embeddings.items():
        (x_train, y_train), (x_test, y_test) = dataloader.load_HPC_log_splitted(train_file, test_file, target_file)
        benchmark_results += run_model('logsy', x_train, y_train, x_test, y_test)

    pd.DataFrame(benchmark_results, columns=['Log_Type', 'Embedding_Type', 'Model', 'Train_Ratio', 'Threshold',
                                             'Accuracy', 'Precision', 'Recall', 'F1']) \
        .to_csv('benchmark_result.csv', index=False)
