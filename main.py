import os
import numpy as np
import extractor
import cluster
import bow
import classifier

from sklearn.model_selection import train_test_split

# CONSTANT
EXAMPLE_IMG = os.path.join("dataset","Multiple","cannon","image_0005.jpg")
DIR_DATA = "data"
DIR_FEATURES = os.path.join(DIR_DATA,"features","Multiple")
DIR_BOW = os.path.join(DIR_DATA, "bow", "Multiple")
DIR_SAMPLES = os.path.join(DIR_DATA,"samples","Multiple")
DIR_DATASET = os.path.join("dataset","Multiple")
NAME_DEFINE = "define_FaceNot.txt"
NAME_TEST_LABEL = "test_label.txt"
NAME_TEST_DIR = "test_dir.txt"
NAME_TRAIN_LABEL = "train_label.txt"
NAME_TRAIN_DIR = "train_dir.txt"
NAME_ALL_FEATURES = "all_features_Multiple.npy"
NAME_ALL_CENTERS = "all_centers_Multiple.npy"
NAME_RESULT = "result.txt"
NAME_SCORE = "score_Multiple.txt"
VALUE_SAMPLES = 5
# CONSTANT - END

# PREVIOUS FUNCTIONS
def define_label():
    file_define = open(os.path.join(DIR_DATA,NAME_DEFINE),"w")
    list_dir = os.listdir(DIR_DATASET)
    for each_dir in range(0,len(list_dir)):
        list_img = os.listdir(os.path.join(DIR_DATASET,list_dir[each_dir]))
        for each_img in range(0, len(list_img)):
            file_define.write(str(os.path.join(DIR_DATASET,list_dir[each_dir],list_img[each_img]))+"\t"+str(list_dir[each_dir])+"\n")
    file_define.close()
def split_train_test(samples=1, test_size=0.2):

    for each_sample in range(samples):
        dir_sample = os.path.join(DIR_SAMPLES,str(each_sample))

        os.makedirs(dir_sample)

        file_define = open(os.path.join(DIR_DATA,NAME_DEFINE),"r")
        list_define = file_define.read().splitlines()

        list_dir_img = []
        list_label_name = []

        for each_define in list_define:
            list_dir_img.append(each_define.split("\t")[0])
            list_label_name.append(each_define.split("\t")[1])
        list_train_dir_img, list_test_dir_img, list_train_label_name, list_test_label_name = train_test_split(list_dir_img, list_label_name, test_size = test_size)

        file_test_label = open(os.path.join(dir_sample,NAME_TEST_LABEL),"w")
        file_test_dir = open(os.path.join(dir_sample,NAME_TEST_DIR),"w")
        file_train_label = open(os.path.join(dir_sample,NAME_TRAIN_LABEL),"w")
        file_train_dir = open(os.path.join(dir_sample,NAME_TRAIN_DIR),"w")

        for each_test_label in list_test_label_name:
            file_test_label.write(each_test_label + "\n")
        for each_test_dir in list_test_dir_img:
            file_test_dir.write(each_test_dir + "\n")

        for each_train_label in list_train_label_name:
            file_train_label.write(each_train_label + "\n")
        for each_train_dir in list_train_dir_img:
            file_train_dir.write(each_train_dir + "\n")

        file_test_label.close()
        file_test_dir.close()
        file_train_label.close()
        file_train_dir.close()
def extract_all_sift_features():
    loading = 0
    list_dir = os.listdir(DIR_DATASET)
    all_sifts = []
    for each_dir in range(0,len(list_dir)):
        loading += 1
        if not os.path.exists(os.path.join(DIR_FEATURES, list_dir[each_dir])):
            os.makedirs(os.path.join(DIR_FEATURES, list_dir[each_dir]))
        list_img = os.listdir(os.path.join(DIR_DATASET, list_dir[each_dir]))
        for each_img in range(0,len(list_img)):
            loading += 1
            print("(",round(loading/1740 * 100,2),"%) : ",list_dir[each_dir], "/", list_img[each_img])
            dir_img = os.path.join(DIR_DATASET, list_dir[each_dir], list_img[each_img])
            descs = extractor.sift_features(dir_img)
            if descs is None:
                continue
            all_sifts.extend(descs)
            np.save(os.path.join(DIR_FEATURES,list_dir[each_dir],list_img[each_img]), descs)
    np.save(os.path.join(DIR_FEATURES,NAME_ALL_FEATURES),all_sifts)
    print ("Length of all sifts: ",len(all_sifts))
def cluster_all_description():
    np.save(os.path.join(DIR_FEATURES, NAME_ALL_CENTERS), cluster.kmeans(os.path.join(DIR_FEATURES, NAME_ALL_FEATURES)))
def build_bow():
    list_dir = os.listdir(DIR_DATASET)
    for each_dir in range(0,len(list_dir)):
        if not os.path.exists(os.path.join(DIR_BOW, list_dir[each_dir])):
            os.makedirs(os.path.join(DIR_BOW, list_dir[each_dir]))
        list_img = os.listdir(os.path.join(DIR_DATASET,list_dir[each_dir]))
        for each_img in range(0,len(list_img)):
            print("//", list_img[each_img])
            centeroids = bow.centeroid(os.path.join(DIR_FEATURES,NAME_ALL_CENTERS))
            bow.build_vector_10k(list_dir[each_dir], list_img[each_img], centeroids)
def predictor():
    list_samples_dir = os.listdir(DIR_SAMPLES)
    svm_mean_score = 0
    file_score = open(os.path.join(DIR_DATA, NAME_SCORE), "w")
    for each_sample in range(0,len(list_samples_dir)):
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        file_train_dir = open(os.path.join(DIR_SAMPLES,list_samples_dir[each_sample],NAME_TRAIN_DIR))
        list_train_dir = file_train_dir.read().splitlines()
        for each_train_dir in range(0, len(list_train_dir)):
            file_train_feature_dir = os.path.join(DIR_DATA, "bow", list_train_dir[each_train_dir][8:]+".npy")
            print(file_train_feature_dir)
            train_features.append(np.load(file_train_feature_dir))

        #print(train_features)
        #print(len(train_features))
        #train_features = np.asarray(train_features)
        #print(train_features.shape)
        #train_features = train_features.reshape((train_features.shape[0], train_features.shape[2]))
        #print(train_features.shape)

        file_train_label = open(os.path.join(DIR_SAMPLES,list_samples_dir[each_sample],NAME_TRAIN_LABEL))
        list_train_label = file_train_label.read().splitlines()

        train_labels = np.array(list_train_label)
        print(train_labels.shape)
        train_labels = train_labels.reshape((1, train_labels.shape[0])).T
        train_labels = np.ravel(train_labels)
        print(train_labels.shape)

        file_test_dir = open(os.path.join(DIR_SAMPLES,list_samples_dir[each_sample],NAME_TEST_DIR))
        list_test_dir = file_test_dir.read().splitlines()
        for each_test_dir in range(0, len(list_test_dir)):
            file_test_feature_dir = os.path.join(DIR_DATA, "bow", list_test_dir[each_test_dir][8:]+".npy")
            test_features.append(np.load(file_test_feature_dir))

        #test_features = np.asarray(test_features)
        #test_features = test_features.reshape((test_features.shape[0], test_features.shape[2]))

        file_test_label = open(os.path.join(DIR_SAMPLES,list_samples_dir[each_sample],NAME_TEST_LABEL))
        list_test_label = file_test_label.read().splitlines()

        test_labels = np.array(list_test_label)
        test_labels = test_labels.reshape((1, test_labels.shape[0])).T
        test_labels = np.ravel(test_labels)

        svm_model = classifier.svm(train_features, train_labels)
        svm_predict_labels = svm_model.predict(test_features)
        svm_score = svm_model.score(test_features, test_labels)
        svm_mean_score += svm_score

        file_predict_result = open(os.path.join(DIR_SAMPLES, list_samples_dir[each_sample], NAME_RESULT), "w")
        for each_label in svm_predict_labels:
            file_predict_result.write(str(each_label)+"\n")
        file_predict_result.close()

        file_score.write("Sample"+str(each_sample)+"\t:"+str(svm_score)+"\n")
        print("Sample"+str(each_sample)+"\t:"+str(svm_score))

    svm_mean_score = svm_mean_score/VALUE_SAMPLES
    file_score.write("------------\n"+"Mean \t:"+str(svm_mean_score))
    file_score.close()



# PREVIOUS FUNCTIONS - END

def main():
    # previous function, do not need to re run.
    # desc: build a define list (img directory \t img label) and 5 samples for testing
    #define_label()
    #split_train_test(samples = VALUE_SAMPLES)
    #extract_all_sift_features()
    #cluster_all_description()
    #build_bow()
    predictor()
    print()

main()
