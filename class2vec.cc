#include<string>
#include<fstream>
#include<cmath>
#include<vector>
using namespace std;

struct Record {
    string code;
    vecotr<int> nodes;
    vector<int> words;
};

int vec_size = 100;
int vocab_size;
int class_num;
float *syn0, *syn1, *neu, *eneu;
float *expTable;
clock_t start;
float starting_alpha = 0.025;
void init_net() {

    syn0 = new float[vec_size * vocab_size];
    syn1 = new float[vec_size * class_num * 2 - 1]();
    neu = new float[vec_size];
    eneu = new float[vec_size];

    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    if (expTable == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    if (!(syn0 && syn1 && neu && eneu)) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < vec_size; ++j) {
            syn1[i * vec_size + j] = (rand() / (float)RAND_MAX - 0.5) / vec_size;
        }
    }
}

void train_model() {
    vector<int> codes;
    vector<int> nodes;
    vector<int> words;
    int record_count = 0;
    float alpha;
    while (1) {
        // read nodes
        // read words;
        alpha = starting_alpha * (1 - record_count / (float)100000);
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        for (int i = 0; i < vec_size; ++i) {
            neu[i] = 0;
            eneu[i] = 0;
        }
        for (int i = 0; i < words.size(); ++i) {
            for (int j = 0; j < vec_size; ++j) {
                neu[j] += syn0[words[i] * vec_size + j];
            }
        }
        for (int i = 0; i < nodes.size(); ++i) {
            float f = 0;
            float g;
            for (int j = 0; j < vec_size; ++j) {
                f += neu[j] * syn1[nodes[i] * vec_size + j];
            }
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            g = (1 - codes[i] - f) * alpha;
            for (int c = 0; c < vec_size; c++) {
                eneu[c] += g * syn1[nodes[i] * vec_size + c];
                syn1[nodes[i] * vec_size + c] += g * neu[c];
            }
        }
        for (int i = 0; i < words.size(); i++) {
            for (int c = 0; c < vec_size; c++) {
                syn0[c + words[i] * vec_size] += enue[c];
            }
        }
    }
}
