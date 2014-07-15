float *update(float *vec1, float *vec2, int dim, float rate) {
    for (int i = 0; i < dim; ++i) {
        vec1[i] += rate * vec2[i];
    }
    return vec1;
}

float product(float *vec1, float *vec2, int dim) {
    float p = 0;
    for (int i = 0; i < dim; ++i) {
        p += vec1[i] * vec2[i];
    }
    return p;
}


float sum(float *vec, int dim) {
    float s = 0;
    for (int i = 0; i < dim; ++i) {
        s += vec[i];
    }
    return s;
}

float *normalize(float *vec, int dim) {
    float sq = product(vec, vec, vec_size);
    for (int i = 0; i < dim; ++i) {
        vec[i] /= sq;
    }
    return vec;
}

float *W;
float *T;
int voc_size;
int topic_num;
int vec_size;
float *C;
float *Ww;
float *Wt;
float lambda;
float alpha;


int max_rount;

void update_words() {
   for (int w = 0; w < vocab_size; ++w) {
       for (int t = 0; t < topic_num; ++t) {
           update(W+ w * vec_size, T + t * vec_size, vec_size, alpha * Ww[w * topic_num + t]);
       }
       normalize(W + w *vec_size, vec_size);
   }
}
void update_topic() {
    for (int t = 0; t < class_num; ++t) {
        for (int w = 0; w < vocab_size; ++w) {
            update(T + t * vec_size, W + w * vec_size, vec_size, alpha * Wt[t * vocab_size + w]);
        }
        normalize(T + t * vec_size, vec_size);
        for (int p = 0; p < class_num; ++p) {
            update(T + t * vec_size, T + t * vec_size, vec_size, alpha * lambda);
        }
        normalize(T + t * vec_size, vec_size);
    }
}

void train () {
    int round = 0;
    while (round < max_rount) {
        update_words();
        update_topic();
        round++;
    }
}
