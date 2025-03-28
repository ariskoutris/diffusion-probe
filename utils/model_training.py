import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
from tqdm.auto import tqdm
import tensorflow as tf
import gc

def compute_embeddings(img_arrs, batch_size=16):
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    vgg16 = VGG16(weights="imagenet", include_top=True)
    features_model = Model(inputs=vgg16.input, outputs=vgg16.get_layer("fc2").output)
    
    n_samples = len(img_arrs)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    feature_dim = 4096  # fc2 layer dimension
    all_features = np.zeros((n_samples, feature_dim))
    
    for i in tqdm(range(n_batches), desc="Computing embeddings"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_imgs = img_arrs[start_idx:end_idx]
        xs_raw = np.stack(batch_imgs)
        xs = preprocess_input(xs_raw)
        
        batch_features = features_model.predict(xs, verbose=0)
        all_features[start_idx:end_idx] = batch_features

        del xs_raw, xs, batch_features
        gc.collect()    

    del features_model, vgg16
    gc.collect()
    
    return all_features


def prepare_train_test_data(vecs, metadata_df, depth1_only=True, encode_labels=True):
    """
    Prepare training and testing data splits.
    If depth1_only=True, use cat_depth_1 as target, otherwise use cat_depth_2.
    Returns x_train, y_train, x_test, y_test, train_df, test_df, and encoder
    """
    # For classification (depth 1 -> depth 1)
    if depth1_only:
        train_df = metadata_df[~metadata_df.cat_depth_1.isin(['None']) & metadata_df.cat_depth_2.isin(['None'])]
        test_df = metadata_df[~metadata_df.cat_depth_1.isin(['None']) & ~metadata_df.cat_depth_2.isin(['None'])]
        target_col = 'cat_depth_1'
    # For clustering (depth 2 labels)
    else:
        test_df = metadata_df[~metadata_df.cat_depth_1.isin(['None']) & ~metadata_df.cat_depth_2.isin(['None'])]
        train_df = metadata_df[metadata_df.cat_depth_1.isin(test_df.cat_depth_1.unique()) & metadata_df.cat_depth_2.isin(['None'])]
        target_col = 'cat_depth_2'
    
    x_train = vecs[train_df.index.values]
    x_test = vecs[test_df.index.values]
    y_train = train_df.cat_depth_1.values.reshape(-1, 1)
    y_test = test_df[target_col].values.reshape(-1, 1)
    
    if encode_labels:
        enc = OrdinalEncoder()
        y_train = enc.fit_transform(y_train).ravel()
        y_test = enc.transform(y_test).ravel() if depth1_only else enc.fit_transform(y_test).ravel()
    else:
        enc = None
    
    return x_train, y_train, x_test, y_test, train_df, test_df, enc


def train_classifier(x_train, y_train, x_test, n_components=6, n_neighbors=71):
    """
    Train a classifier using pre-split data.
    
    Parameters:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        n_components: Number of PCA components
        n_neighbors: Number of neighbors for KNN
        
    Returns:
        clf: Trained classifier
        pca: Fitted PCA model
        x_train_nd: PCA-transformed training data
        x_test_nd: PCA-transformed test data
        preds_train: Predictions on training data
        preds_test: Predictions on test data
    """
    # Reduce dimensions using PCA
    pca = PCA(n_components=n_components)
    x_train_nd = pca.fit_transform(x_train)
    x_test_nd = pca.transform(x_test)
    
    # Train KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train_nd, y_train)
    preds_train = clf.predict(x_train_nd)
    preds_test = clf.predict(x_test_nd)

    return clf, pca, x_train_nd, x_test_nd, preds_train, preds_test