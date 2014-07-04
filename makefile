all: train predict
train: class2vec.cc
	g++ class2vec.cc -o train
predict: class_predict.cc
	g++ class_predict.cc -o predict
clean:
	rm train predict
