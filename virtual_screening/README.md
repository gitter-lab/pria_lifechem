# Related Works

- DeepTox: Toxicity Prediction using Deep Learning
    - all ReLU layers except the output layer using sigmoid, cross-entropy as loss function(logistic loss)
    - to cope with missing values, set weight as 0 if the label is missing, otherwise is 1
    - SGD
    - \# layers {1,2,3,4}, \# units {1024,2048,4096,8192,16384}
    - AUC as early stopping
    - tuning for multitask, and also each single task
    - 10 out of 12 essays, multitask outperformed single task
    - doesn't tell which hperparameter set works best for multitask


- Deep Learning as an Opportunity in Virtual Screening:
    - compared MTNN with other ML and Bio methods, no STNN
    - simply extend the work by Merk Kaggle Challenge by changing dataset to ChEMBL, because Tox21(Merk Kaggle's dataset) has small data scales
    - for not active label, multiply corresponding loss function by 0
    - give weights to each task by amount of available data, in order to make sure they have same impact on each layer
    - from the above two, I guess this ChEMBL dataset only has active values
    - didn't compare MTNN with STNN


- Massively Multitask Networks for Drug Discovery
    - Single Task Nerural Net
    - Pyramidal (2000,100) STNN
    - 1-Hidden (1200) Layer Multitask Neural Net
    - Pyramidal (2000,100) Multitask Neural Net
    - more tasks or more data:
    - try different task number, {10,15, 20, 30, 50, 82}
    - try different datapoints, {1.6M, 3.3M, 6.5M, 13M}
    - both improve AUC, only a little
    - didn't mention how they treat missing valuse, also no class weights showed up
    - \> 1M epoches


- Modeling Industrial ADMET with Multitask Networks
    - STNN, MTNN, W-MTNN(3.1 assign weights to each task)
    - ReLU activations, batch normalizer, learning rate 0.001, batch size 128, 1M epoches
    - set weight to 0 for missing data
    - hidden units: (1000), (2000,100), (2000,1000), (4000,2000,1000,1000), (4000)
    - AUC, and select best models during training. also use enrichment scores
    - temporal validation vs. random cross-validation (LB1, LB2, LB3 have temporal relation?)
    - info leaky: training set for one task is unrealistically related to training set for another task


- Low Data Drug Discovery with One-shot Learning
    - one-shot learning
    - Residual LSTM
    - training on some tasks, and test on other tasks
    - no mention of missing values
    - main contribution is building up a pipeline, DeepChem
    
- Learning to SMILE(S)
    - textual representation, SMILES
    - and two baseline fingerprint comopund, MACCS and KRFP
    - model comparison, SVM, NB, RF, CNN, RNNLM(RNN Language Model)
    - Log-loss
    - 5 sampled targetes
    - Showed RNN and CNN can outperform classical classifier, but no comparison with simple NN
    
- DeepCCI:End-to-end Deep Learning for Chemical-Chemical Interaction Prediction
    - Chemical-chemical interaction (CCI)
    - SMILES as feature
    - takes SMILES strings as inputs for two objects, and produce interaction probability
    - applies CNN
    - no context information for SMILES embedding
    - CCI data, downloaded from STITCH
    - evaluation: ROC AUC, ACC, TPR, TNR, PPV, NPV, F1
    
- Grammar Variational Autoencoder
    - extract grammar feature from a parsing tree, which is built based on SMILES
    - compare grammar-level LSTM to SMILES-level LSTM
    
- IRV:
    - Influence Relevance Voting: An Accurate And Interpretable Virtual High Throughput Screening Methods
    - Accurate and efficient target prediction using a potency-sensitive influence-relevance voter
    
 - Other
    - The Relationship Between Precision-Recall and ROC Curves: PR curve has interpolation problem, we solve it by R package instead of sklearn
    - The Marginal Value of Adaptive Gradient Methods in Machine Learning: Adaptive method like Adam, cannot perform better than SGD


All of them use median AUC as evaluation metrics, and stanford group also uses enrichment scores. But as the paper The Relationship Between Precision-Recall and ROC Curves demonstrates, PR can be a better metrics, and maybe we can use this to give another explanation for stanford group's dataset-dependent result. And also using enrichment factor can help.

## Other References

NIPS 2013, Multi-Task Bayesian Optimization

NIPS 2012, Practical Bayesian Optimization of Machine Learning Algorithms

NIPS 2014，Generative Adversarial Nets

ArXiv 2016, Matching Networks for One Shot Learning

ICLR 2016, Order Matters: Sequence To Sequence For Sets

ArXiv 2016, Not Just A Black Box: Learning Important Features Through Propagating Activation Differences

NIPS 2016, Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Network

ICLR 2015, Adam: A Method For Stochastic Optimization

# data_preparation

## split pcba into folds

    Algorithm:
        1- shuffle the pcba rows randomly
        2- sort the pcba labels from smallest active_counts to largest
        3- iterate on this sorted label list and do:
            -create k folds which will contain the row indexes only
            -split the active_indexes into the k folds
            -split the inactives_indexes into the k folds
            -split the missing_indexes into the k folds
            
            -uniquify each fold to remove duplicate row indexes.
            
            -greedily remove overlapping indexes from each fold. start with 
             fold 0 and remove from the other 1-k folds. then fold 1 and remove
             from the other 2-k folds. then fold 2 and remove from the other 
             3-k folds. and so on. This ensures that the top most fold contains
             the row index and all other folds do not.
        
        4- uniquify each fold to remove duplicate row indexes just in case
