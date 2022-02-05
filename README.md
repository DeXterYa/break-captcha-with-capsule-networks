# Breaking CAPTCHA with Capsule Networks

### Folder structure
Data generation folders:
* **captcha_generator**: Generate "Dataset 1" mentioned in the paper.
* **captcha_generator_split**: Generate "Dataset 2" mentioned in the paper.


* **cnn_captcha**: The implementation of Deep-CAPTCHA model used to classify "Dataset 1".
* **dynamic_captcha**: The implementation of CapsNet model used to classify "Dataset 1".


* **cnn_captcha_split**: The implementation of Deep-CAPTCHA model used to classify "Dataset 2".
* **dynamic_captcha_split**: The implementation of Deep-CAPTCHA model used to classify "Dataset 2".


### Experiments
4.1 Adapt the CapsNet to the CAPTCHA dataset

1. Run `python captcha_generator.py` inside folder **captcha_generator** to generate "CAPTCHA_4digits_noise" (Dataset 1)
2. Copy the folder "CAPTCHA_4digits_noise" to `cnn_captcha/data` or `dynamic_captcha/data` as the dataset for those two models
3. Configure the script in `cnn_captcha/local_experiment_scripts` or `dynamic_caps_captcha/local_experiment_scripts`
4. Use `bash cnn_captcha4digit_test.sh` or `bash dynamic_captcha4digit_test.sh` to run training jobs. The results will be stored in a folder with the name specified in the script.



4.2 Adapt CapsNet to the CAPTCHA Puzzle Task

1. Run `python captcha_generator.py` inside folder **captcha_generator_split** to generate "CAPTCHA_3digits_noise" (Dataset 2)
2. Copy the folder "CAPTCHA_3digits_noise" to `cnn_captcha_split/data` or `dynamic_caps_captcha_split/data` as the dataset for those two models
3. Configure the script in `cnn_captcha_split/local_experiment_scripts` or `dynamic_caps_captcha_split/local_experiment_scripts`
4. Use `bash cnn_captcha4digit_test.sh` or `bash dynamic_captcha4digit_test.sh` to run training jobs. The results will be stored in a folder with the name specified in the script.
