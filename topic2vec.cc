#include<cstdlib>
#include<cmath>
#include<iostream>
#include<ctime>
using namespace std;

float *W;
float *T;
int vocab_size;
int topic_num;
int vec_size;
int *C;
float *Ww;
float *Wt;
float lambda;
float alpha;


int max_round;

float *update(float *vec1, float *vec2, float rate) {
    for (int i = 0; i < vec_size; ++i) {
        vec1[i] += rate * vec2[i];
    }
    return vec1;
}

float product(float *vec1, float *vec2) {
    float p = 0;
    for (int i = 0; i < vec_size; ++i) {
        p += vec1[i] * vec2[i];
    }
    return p;
}


float sum(float *vec) {
    float s = 0;
    for (int i = 0; i < vec_size; ++i) {
        s += vec[i];
    }
    return s;
}

float *normalize(float *vec) {
    float sq = sqrt(product(vec, vec));
    for (int i = 0; i < vec_size; ++i) {
        vec[i] /= sq;
    }
    return vec;
}

float cosine (float *vec1, float *vec2) {
	float p = product(vec1, vec2);
	float r1 = product(vec1, vec1);
	float r2 = product(vec2, vec2);
	return p / sqrt(r1) / sqrt(r2);
}

void update_words() {
   for (int w = 0; w < vocab_size; ++w) {
       for (int t = 0; t < topic_num; ++t) {
       	   float b = exp(product(T + t * vec_size, W + w * vec_size));
           update(W+ w * vec_size, T + t * vec_size, alpha * b * Ww[w * topic_num + t]);
       }
       //normalize(W + w *vec_size);
   }
}
void update_topic() {
    for (int t = 0; t < topic_num; ++t) {
        for (int w = 0; w < vocab_size; ++w) {
        	float b = exp(product(T + t * vec_size, W + w * vec_size));
            update(T + t * vec_size, W + w * vec_size, alpha * b * Wt[t * vocab_size + w]);
        }
        //normalize(T + t * vec_size);
        for (int p = 0; p < topic_num; ++p) {
        	if (p == t) continue;
        	float b = exp(product(T + t * vec_size, T + p * vec_size));
            update(T + t * vec_size, T + t * vec_size, alpha * lambda * b);
        }
        //normalize(T + t * vec_size);
    }
}

void train() {
    int round = 0;
    while (round < max_round) {
        update_words();
        update_topic();
        round++;
    }
}

void init() {
	W = new float[vocab_size * vec_size];
	T = new float[topic_num * vec_size];
	C = new int[vocab_size * topic_num];
	Ww = new float[vocab_size * topic_num];
	Wt = new float[topic_num * vocab_size];
}

void test() {
	//topic_num = 4;
	//vec_size = 2;
	vocab_size = 1;
	init();
	for (int i = 0; i < topic_num; i++) {
		for (int j = 0; j < vec_size; j++) {
			T[i * vec_size + j] = rand() * 1.0 / RAND_MAX;
		}
		//normalize(T + i * vec_size);
	}
	alpha = 0.025;
	max_round = 10000000;
	int round =  0;
	while (round++ < max_round) {
		for (int i = 0; i < topic_num; i++) {
			for (int j = 0; j < topic_num; j++) {	
				if (i == j) continue;
				float p = product(T + i * vec_size, T + j * vec_size);
				p = exp(p);
				float rate = alpha * (1 - round / (float)100000);
				rate = rate < 0.00001 ? 0.00001 : rate;
				rate = -rate * p;
				update(T + i * vec_size, T + j * vec_size, rate);
			}
			//normalize(T + i * vec_size);
		}
	}
	float P = 0;
	for (int i = 0; i < topic_num; i++) {
		for (int j = 0; j < topic_num; j++) {
			if (i == j) continue;
			P += max((float)0.0, product(T + vec_size * i, T + vec_size * j));
			cout << acos(cosine(T + vec_size * i, T + vec_size * j)) * 180 / 3.1415926 
				<< endl;
		}
		cout << endl;
	}
	cout << P << endl;
	for (int i = 0; i < topic_num; i++) {
		cout << sqrt(product(T + vec_size * i, T + vec_size * i)) << endl;
	}
}

int main(int argc, char *argv[]) {
	topic_num = atoi(argv[1]);
	vec_size = atoi(argv[2]);
	srand((unsigned int) time(NULL));
	test();
	return 0;
}
