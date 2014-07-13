#include<string>
#include<cstring>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<vector>
#include<map>
#include<set>
#include<cassert>
using namespace std;
#define MAX_CODE_LEN 60
#define MAX_RECORD_WORDS 60
#define MAX_WORD_LEN 60
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
// map code to its index in the syn1
map<string, int> code_index;

// parent[i] is the index in syn1 of parent of the node who's index in syn1 is i
int *parent;
int *code_table;
//int *children;
//string *node_name;
// map vocab word to its index in the syn0
map<string, int> vocab_index;
//int latest_code_index = 0;
// the size of each vector
int vec_size = 100;
// how many vocab word in the table
int vocab_size;
// how many class
int node_num;
// if a word appears less than min_word_count, it will be ignored
int min_word_count = 3;
// if a reocrd has less than min_record_words words, it will be ignored
int min_record_words = 3;
// syn0: vector table for word
// syn1: vector table for class
// neu:  vecotr for middle sum unit
// eneu: error vector for middle sum unit
float *syn0, *syn1, *neu, *eneu;
// for fast compute 1/(1 + exp(x))
float *expTable;
// inital parameter changing step
float starting_alpha = 0.025;
float beta = 0.05;
int max_round = 5;
struct Tree {
    string name;
    int index;
    struct Tree *link[2];
    Tree (string n) : name(n) {link[0] = 0; link[1] = 0;}
};

Tree *code_tree;

Tree *add_to_tree(Tree *root, string name) {
    Tree *p = root->link[0];
    if (p == 0) {
        p = new Tree(name);
        root->link[0] = p;
        return p;
    }
    while (p->name != name && p->link[1] != 0) {
        p = p->link[1];
    }
    if (p->name == name) return p;
    p->link[1] = new Tree(name);
    return p->link[1];
}

int ajust_tree(Tree *root) {
	// no child
	if (root == 0) return 0;
    if (root->link[0] == 0) {
    	root->link[1] = 0;
    	return 1;
	}
    Tree *p1 = root->link[0], *p2 = root->link[0];
    // one child
    if (p1->link[1] == 0) {
        root->name = root->name == "*" ? p1->name : root->name + '.' + p1->name;
        root->link[0] = p1->link[0];
        delete p1;
        return ajust_tree(root);
    }
    // two children
    if (p1->link[1]->link[1] == 0) {
        root->link[1] = p1->link[1];
        p1->link[1] = 0;
    } else if (p1->link[1]->link[1]->link[1] == 0) {
    // three children
    	root->link[1] = p1->link[1]->link[1];
        root->link[0] = new Tree("*");
        root->link[0]->link[0] = p1;
        p1->link[1]->link[1] = 0;
    } else {
		// more than three
		root->link[0] = new Tree("*");
		root->link[0]->link[0] = p1;
		root->link[1] = new Tree("*");
		while (p2 && p2->link[1]) {
			p1 = p1->link[1];
			p2 = p2->link[1]->link[1];
		}
		root->link[1]->link[0] = p1->link[1];
		p1->link[1] = 0;
	}
    return 1 + ajust_tree(root->link[0]) + ajust_tree(root->link[1]);
}

void map_code_to_index(Tree *t, string prefix) {
	if (t == 0) return;
	int begin = 0, end;
	while (begin < t->name.length()) {
		end = begin;
		while (end < t->name.length() && t->name[end] != '.') end++;
		if (t->name.substr(begin, end - begin) == "*") {
			begin = end + 1;
			continue;
		}
		prefix = prefix == "" ? t->name.substr(begin, end - begin) : 
			prefix + '.' + t->name.substr(begin, end - begin);
		code_index[prefix] = t->index;
		begin = end + 1;
	}
	map_code_to_index(t->link[0], prefix);
	map_code_to_index(t->link[1], prefix);
}
// read a word from fin
// return: 0 if '\n'
//         len if read a word
//        -1 if end of file
int read_word(FILE *fin, char *buffer) {
    int len = 0;
    char ch;
    if (feof(fin)) {
        return -1;
    }
    while ((ch = fgetc(fin)) != EOF) {
        if (ch == '\n') {
            if (len > 0) {
                ungetc(ch, fin);
            }
            break;
        } else if (ch == ' ' || ch == '\t') {
            if (len == 0) continue;
            break;
        } else if (len < MAX_WORD_LEN - 1) {
            buffer[len++] = ch;
        }
    }
    buffer[len] = '\0';
    if (len == 0 && ch == EOF) return -1;
    return len;
}

int assign_node_index(Tree *t, int index) {
    t->index = index;
    if (t->link[0] == 0) return index + 1;
    int next = assign_node_index(t->link[0], index + 1);
    parent[index + 1] = index;
    code_table[index + 1] = 0;
    parent[next] = index;
    code_table[next] = 1;
    return assign_node_index (t->link[1], next);
}
void learn_vocab(FILE *train_file) {
    char word[MAX_WORD_LEN];
    int word_len;
    map<string, int> word_count; // count each number for filter words
    set<string> codes;

    fseek(train_file, 0, SEEK_SET);
    int new_record = 1;
    while((word_len = read_word(train_file, word)) != -1) {
        if (word_len == 0) {
            new_record = 1;
        } else if (new_record) {
            codes.insert(word);
            new_record = 0;
        } else {
            word_count[word]++;
        }
    }

    int word_next_index = 0;
    for (map<string, int>::iterator it = word_count.begin(); it != word_count.end(); ++it) {
        if (it->second >= min_word_count) {
            vocab_index[it->first] = word_next_index++;
        }
    }
    vocab_size = vocab_index.size();

    code_tree = new Tree("*");
    for (set<string>::iterator it = codes.begin(); it != codes.end(); ++it) {
        int begin = 0, end;
        Tree *t = code_tree;
        while (begin < it->length()) {
        	end = begin;
            while ((*it)[end] != '.' && end != it->length()){
                end++;
            }
            string name = it->substr(begin, end - begin);
            t = add_to_tree(t, name); 
            begin = end + 1;
        }
    }
    node_num = ajust_tree(code_tree);
    parent = new int[node_num]();
    code_table = new int[node_num]();
    assign_node_index(code_tree, 0);
    map_code_to_index(code_tree, "");
}

void init_net() {
    syn0 = new float[vec_size * vocab_size];
    syn1 = new float[vec_size * node_num]();
    neu = new float[vec_size];
    eneu = new float[vec_size];
    expTable = new float[EXP_TABLE_SIZE + 1];
    if (!(syn0 && syn1 && neu && eneu && expTable)) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    for (int i = 0; i < node_num; ++i) {
        for (int j = 0; j < vec_size; ++j) {
            syn1[i * vec_size + j] = (rand() / (float)RAND_MAX - 0.5) / vec_size;
        }
    }
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < vec_size; ++j) {
            syn0[i * vec_size + j] = (rand() / (float)RAND_MAX - 0.5) / vec_size;
        }
    }
}

int read_record(FILE *fin, int *code, int *nodes, int *words, int &len_code, int &len_words) {
    char buffer[MAX_WORD_LEN];
    len_code = 0;
    len_words = 0;
    int len;
    int read_code = 1;
    while ((len = read_word(fin, buffer)) != -1) {
        if (len == 0) {
            if (read_code == 0) break;
        } else if (read_code) {
        	int node = code_index[buffer];
        	// NOTE: the first node is fixed to 0 but not cotained in nodes
        	// The last element in nodes is CLASS VECTOR, which shouldn't be
        	// used to predict!!
        	while (parent[node]) {
        		code[len_code] = code_table[node];
        		nodes[len_code] = node;
        		node = parent[node];
        		len_code++;
			}
			nodes[len_code] = 0;
			reverse(code, code + len_code);
			reverse(nodes, nodes + len_code + 1);

            read_code = 0;
        } else if (vocab_index.count(buffer) > 0 && len_words < MAX_RECORD_WORDS) {
            words[len_words] = vocab_index[buffer];
            ++len_words;
        }
    }
    return len_code > 0;
}

void train_model(FILE *train_file) {
    int code[MAX_CODE_LEN];
    int nodes[MAX_CODE_LEN];
    int words[MAX_RECORD_WORDS];
    int len_code, len_words;
    int record_count = 0;
    float alpha;
    fseek(train_file, 0, SEEK_SET);
    while (read_record(train_file, code, nodes, words, len_code, len_words )) {
        if (len_words < min_record_words) continue;
        //cerr << record_count << endl;
        alpha = starting_alpha * (1 - record_count / (float)100000);
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        for (int i = 0; i < vec_size; ++i) {
            neu[i] = 0;
            eneu[i] = 0;
        }
        for (int i = 0; i < len_words; ++i) {
            for (int j = 0; j < vec_size; ++j) {
                neu[j] += syn0[words[i] * vec_size + j];
            }
        }
        for (int i = 0; i < len_code; ++i) {
            float f = 0;
            float g;
            for (int j = 0; j < vec_size; ++j) {
                f += neu[j] * syn1[nodes[i] * vec_size + j];
            }
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            g = (1 - code[i] - f) * alpha;
            for (int c = 0; c < vec_size; ++c) {
                eneu[c] += g * syn1[nodes[i] * vec_size + c];
                syn1[nodes[i] * vec_size + c] += g * neu[c];
            }
        }
        for (int i = 0; i < len_words; i++) {
            for (int c = 0; c < vec_size; c++) {
                syn0[c + words[i] * vec_size] += eneu[c];
            }
        }
        // update class vector;
        for (int i = 0; i < vec_size; i++) {
        	int id = nodes[len_code] * vec_size + i;
        	syn1[id] = syn1[id] * (1 - beta) + beta * (neu[i] - eneu[i]);
		}
        ++record_count;
    }
}

void dfs_save_code(FILE *fout, Tree *t, string code = "/", string prefix = "") {
	if (t == 0) return;
	if (code == "/") {
		prefix = t->name;
	} else {
		prefix = prefix + '/' + t->name;
	}
	fprintf(fout, "%s\t%s", code.c_str(), prefix.c_str());

	for (int i = 0; i < vec_size; i++) {
		fprintf(fout, "\t%f", syn1[t->index * vec_size + i]);
	}
	fprintf (fout, "\n");
	dfs_save_code(fout, t->link[0], code + '0', prefix);
	dfs_save_code(fout, t->link[1], code + '1', prefix);
}
void save_model(FILE *vocab_vec_file, FILE *class_vec_file) {
    fprintf(vocab_vec_file, "%d %d\n", vocab_size, vec_size);
    for (map<string, int>::iterator it = vocab_index.begin(); it != vocab_index.end(); ++it) {
        fprintf (vocab_vec_file, "%s\t", it->first.c_str());
        for (int i = 0; i < vec_size; ++i) {
            char sep = i == vec_size - 1 ? '\n' : '\t';
            fprintf(vocab_vec_file, "%f%c", syn0[it->second * vec_size + i], sep);
        }
    }
    char code[MAX_CODE_LEN];
    fprintf(class_vec_file, "%d\n", node_num);
    dfs_save_code(class_vec_file, code_tree);
}
int main (int argc, char *argv[]) {
    FILE *train_file;
    FILE *vocab_vec_file;
    FILE *class_vec_file;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0) vec_size = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-m") == 0) min_word_count = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-r") == 0) max_round = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-n") == 0) min_record_words = atoi(argv[i + 1]);
    }
    train_file = fopen("train.dat", "r");
    vocab_vec_file = fopen("vocab_vec.dat", "w");
    class_vec_file = fopen("node_vec.dat", "w");
    if (!(train_file && vocab_vec_file && class_vec_file)) {
        printf ("Open file error\n");
        exit(1);
    }

    fprintf(stderr, "learning vocabulary\n");
    learn_vocab(train_file);
    fprintf(stderr, "vocab size:\t%d\n", vocab_size);
    init_net();
    for (int i = 0; i < max_round; ++i) {
        fprintf(stderr, "start round %d/%d\n", i + 1, max_round);
        train_model(train_file);
    }
    
    fprintf(stderr, "saving model\n");
    save_model(vocab_vec_file, class_vec_file);
    fclose(train_file);
    fclose(vocab_vec_file);
    fclose(class_vec_file);
    
    fprintf(stderr, "vector size:\t%d\n", vec_size);
    fprintf(stderr, "vocab  size:\t%d\n", vocab_size);
    fprintf(stderr, "node number:\t%d\n", node_num);
    return 0;
}
