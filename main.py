import os
import extractor
import cluster

from sklearn.model_selection import train_test_split

# CONSTANT
EXAMPLE_IMG = os.path.join("images","101_ObjectCategories","cannon","image_0005.jpg")
DIR_DATA = "data"
DIR_DATASET = os.path.join("images","101_ObjectCategories")
NAME_DEFINE = "define.txt"
NAME_TEST_LABEL = "test_label.txt"
NAME_TEST_DIR = "test_dir.txt"
NAME_TRAIN_LABEL = "train_label.txt"
NAME_TRAIN_DIR = "train_dir.txt"
VALUE_SAMPLES = 5
VALUE_CLUSTERS = 20
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
def split_train_test(samples=1, test_size=0.33):
    DIR_SAMPLES = os.path.join(DIR_DATA,"samples")

    for each_sample in range(samples):
        DIR_SAMPLE = os.path.join(DIR_SAMPLES,str(each_sample))

        os.makedirs(DIR_SAMPLE)
        os.chmod(DIR_SAMPLE,0o777)

        file_define = open(os.path.join(DIR_DATA,NAME_DEFINE),"r")
        list_define = file_define.read().splitlines()

        list_dir_img = []
        list_label_name = []

        for each_define in list_define:
            list_dir_img.append(each_define.split("\t")[0])
            list_label_name.append(each_define.split("\t")[1])
        list_train_dir_img, list_test_dir_img, list_train_label_name, list_test_label_name = train_test_split(list_dir_img, list_label_name, test_size = test_size)

        file_test_label = open(os.path.join(DIR_SAMPLE,NAME_TEST_LABEL),"w")
        file_test_dir = open(os.path.join(DIR_SAMPLE,NAME_TEST_DIR),"w")
        file_train_label = open(os.path.join(DIR_SAMPLE,NAME_TRAIN_LABEL),"w")
        file_train_dir = open(os.path.join(DIR_SAMPLE,NAME_TRAIN_DIR),"w")

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
# PREVIOUS FUNCTIONS - END

def main():
    # previous function, do not need to re run.
    # desc: build a define list (img directory \t img label) and 5 samples for testing
    #define_label()
    #split_train_test(samples = VALUE_SAMPLES)

    _,list_feature_vectors = extractor.sift_features(EXAMPLE_IMG)
    cluster.kmeans(list_feature_vectors, VALUE_CLUSTERS)


main()

