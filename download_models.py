import os
import gdown

os.makedirs("models", exist_ok=True)

files_to_download = {
    "cnn_binary_class.h5": "1dcKMFt_U8-LZVNntaTikC-_sRiqG4fLW",
    "cnn_multi_class.h5": "115pErRixdy2xLQw0AE8p4y0GIlXVaDUf",
    "knn_binary_class.sav": "1wFhet2ezB_tov6e62OCvTnJ0DgyVq1nK",
    "knn_multi_class.sav": "1idahKdNa56ht6PTc8Wrgp1OmpHt9sovi",
    "lstm_binary_class.h5": "1Edtpum9impxR4cDDnDnPqz5JZJDSuYEV",
    "lstm_multi_class.h5": "1fWFv-XieelmzqDadCgutEqT1BfscASIo",
    "random_forest_binary_class.sav": "1s5GCn2qQQcLGQIe1TxiQOTdUfUFjZHWP",
    "random_forest_multi_class.sav": "1KwX9XikSATBwQqq4Wv-_L8GbzXj1XNJC",
    "encoder_cnn_binary.joblib": "1PkYjyd6Aq7T6M187wW6ARAJscWd1ud8M",
    "encoder_cnn_multi_class.joblib": "1xV5Rial-TEotuPqFjqu7y7WN6dCV9WMM",
    "encoder_knn_binary.joblib": "19t38qcHa-U8czy0wq1LTtbE-STSuClaY",
    "encoder_knn_multi_class.joblib": "17-A5rmIp_L2T9mhjxBmI8RhFOxZF2beg",
    "encoder_lstm_binary.joblib": "1FxAyzGcM3GFJ5lRZWxCUWwfC08C-k-0c",
    "encoder_lstm_multi_class": "1mveMN1GPJu23Ebcuk07BjAPngSBLlz7q",
    "encoder_random_forest_binary_class.joblib": "1Don3KqOiPvs3s9Re9iGynO6KbmIDhImV",
    "encoder_random_forest_multi_class.joblib": "14u4kBr0r2grE-0_ppo2qryS2oSMCsnyz",
    "label_encoder_cnn_binary.joblib": "1_7VipnYIsONVT6yh_ueqUt3J3W-Ex2bl",
    "label_encoder_cnn_multi_class.joblib": "19bh9qP8guS9iJDliedcy3R7EjbRcWJQ0",
    "label_encoder_knn_binary.joblib": "1aTobBlygpjE2oRplU_-KkgkdFCb8DBwk",
    "label_encoder_knn_multi_class.joblib": "1iry6Z53FHsAx4eKLl0X2JM16IhdG4jrg",
    "label_encoder_lstm_binary.joblib": "1CMsoXDgQo_5m6EYrXsRfx2MHUik4zfFK",
    "label_encoder_lstm_multi_class": "1HTB2VNBJH2y4JLXYM7sFSIQ4dkio56D4",
    "label_encoder_random_forest_binary_class.joblib": "1lCdxT6CPL7IVq0oLZPRUVBtKew4sYgDZ",
    "label_encoder_random_forest_multi_class.joblib": "1Rtb7w-KW1pI2Y8s9Q5R7czgBTzMXxTez",
    "pca_knn_binary.joblib": "1jnLYfB8tBCLncadR2llXqX48YmVIgWEX",
    "pca_knn_multi_class.joblib": "1s5GCn2qQQcLGQIe1TxiQOTdUfUFjZHWP",
    "scalar_knn_binary.joblib": "10SdtyALqznRPXfbmTEUAWRDt7dv5cdI8",
    "scalar_knn_multi_class.joblib": "1Kx5HDWeUfHnJoFZ3Xi45DLRxh7OQncA_",
    "scalar_random_forest_binary.joblib": "1lpBKMle_N7TON_mQuU_pJ2qixouhfA6B",
    "scalar_random_forest_multi_class.joblib": "1puaSv5Cq0FMImtPATdiOcKzaBr7XAZJc"

    
}

for filename, file_id in files_to_download.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"NIDS_models/{filename}"
    gdown.download(url, output, quiet=False)
