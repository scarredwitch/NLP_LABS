## A Fast and Accurate Dependency Parser using Neural Networks

### Authors: Danqi Chen and Christopher D. Manning

| Topic  | A Fast and Accurate Dependency Parser using Neural Networks |
|--------------|--------------------------------------------------------------------------------------------------------|
| Problem | Dependency Parser cannot handle complex real-world sentences in a fast and accurate manner |
| Key Related Work | 1. Koo et al. (2008) demonstrated improved parsing performance through techniques like word class features.<br /> 2. Bohnet (2010) reports inefficiency of his baseline parser in terms of feature extraction. <br /> 3. Collobert et al. (2011) successfully displayed the efficiency of distributed word representations in NLP task like POS tagging.|
| Method |Train a neural network classifier to make parsing decisions within a transition-based dependency parser. Introduce a novel activation function for this neural network that captures higher-order interaction features. |
|  Results | 1. Fast computation while achieving 2% improvement in UAS and LAS on both English and Chinese datasets. <br /> 2. Outperforms other greedy parsers using sparse indicator features in both accuracy and speed.|
| Future work | Authors developed a parser that outperforms current parsers and made a significant contribution to field of NLP. In future, authors hope to combine this neural network based classifier with search based models to further improve accuracy.|